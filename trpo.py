# from myconfig import myconfig
from myconfig import myconfig
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import copy
from tensorflow.python.keras.layers import Input, Dense, concatenate, LeakyReLU, Softmax
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.losses import mean_squared_error
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.ndimage.interpolation import shift
from tensorflow.keras.utils import to_categorical
from collections import deque
# import warnings
from critic import Critic
from utils import conjugate_gradient, set_from_flat, kl, self_kl, \
    flat_gradient, get_flat, discount, line_search, gauss_log_prob, visualize, gradient_summary, \
    unnormalize_action
# from  continuous.gail_atm.critic import Critic
# from continuous.gail_atm.utils import *

import random


# from cartpole.critic.critic import Critic
# from cartpole.trpo_plugins.utils import conjugate_gradient, set_from_flat, kl, self_kl,\
#     flat_gradient, get_flat, discount, line_search
# np.seterr(all='warn')
# warnings.filterwarnings('error')

# http://rail.eecs.berkeley.edu/deeprlcoursesp17/docs/lec5.pdf

# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
#config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.7

class ClonosEncoder(object):

    @staticmethod
    def create_encoder(observation_dimensions, latent_dimensions):
        with tf.name_scope('Encoder'):
            encoder_x = Input(shape=(observation_dimensions+latent_dimensions,), dtype=tf.float32)

            l1 = Dense(100, activation='tanh')(encoder_x) #name='dense_6'), activation=tf.keras.layers.LeakyReLU(alpha=0.01)
            l2 = Dense(100, activation='tanh')(l1) #name='dense_7')

            out = Dense(latent_dimensions, activation='softmax')(l2) #name='dense_8')

        model = Model(inputs=encoder_x, outputs=out)

        return model, encoder_x, l1, l2


class Policy(object):

    @staticmethod
    def create_policy(observation_dimensions, latent_dimensions, action_dimensions):
        """
        Creates the model of the policy.
        :param observation_dimensions: Observations' dimensions.
        :param action_dimensions: Actions' dimensions.
        :return: Model and the Input layer.
        """
        with tf.name_scope('Policy'):
            x = Input(shape=(observation_dimensions+latent_dimensions,), dtype=tf.float32)

            h = Dense(100, activation='tanh')(x)#, name='Policy/dense_9')(x)
            h1 = Dense(100, activation='tanh')(h)#, name='Policy/dense_10')(h)

            out = Dense(action_dimensions)(h1)#, name='Policy/dense_11')(h1)

        model = Model(inputs=x, outputs=out)

        return model, x, h, h1

class Clonos(object):
    # load json and create model

    @staticmethod
    def create_Clonos():
        json_file = open('./VAE/results_gumbel_softmax/model/'+'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_enc_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_enc_model.load_weights('./VAE/results_gumbel_softmax/model/'+'model.h5')
        print("Loaded model from disk")
        loaded_enc_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        #score = loaded_enc_model.evaluate(X, Y, verbose=0)
        #print("%s: %.2f%%" % (loaded_enc_model.metrics_names[1], score[1] * 100))


        return loaded_enc_model


class Discriminator(object):

    def __init__(self, observation_dimensions=11, action_dimensions=3):
        self.alpha = myconfig['discriminator_alpha']
        self.epochs = myconfig['discriminator_epochs']
        self.sess = tf.Session(config=config)

        self.observations_input = Input(shape=(observation_dimensions,), dtype=tf.float32)
        self.actions_input = Input(shape=(action_dimensions,), dtype=tf.float32)
        self.input = concatenate([self.observations_input, self.actions_input])

        h1 = Dense(100, activation='tanh')(self.input)
        h2 = Dense(100, activation='tanh')(h1)

        self.out = Dense(1)(h2)

        self.discriminate = Model(inputs=[self.observations_input, self.actions_input], outputs=self.out)

        self.log_D = tf.log(tf.nn.sigmoid(self.out))

        self.expert_samples_observations = Input(shape=(observation_dimensions,), dtype=tf.float32)
        self.expert_samples_actions = Input(shape=(action_dimensions,), dtype=tf.float32)
        self.policy_samples_observations = Input(shape=(observation_dimensions,), dtype=tf.float32)
        self.policy_samples_actions = Input(shape=(action_dimensions,), dtype=tf.float32)
        self.expert_samples_out = self.discriminate([self.expert_samples_observations, self.expert_samples_actions])
        self.policy_samples_out = self.discriminate([self.policy_samples_observations, self.policy_samples_actions])

        self.discriminator = Model(inputs=[self.expert_samples_observations,
                                           self.expert_samples_actions,
                                           self.policy_samples_observations,
                                           self.policy_samples_actions
                                           ],
                                   outputs=[self.expert_samples_out, self.policy_samples_out]
                                   )

        # self.expert_loss = tf.reduce_mean(tf.log(tf.ones_like(self.expert_samples_out)-tf.nn.sigmoid(self.expert_samples_out)))
        # self.policy_loss = tf.reduce_mean(tf.log(tf.nn.sigmoid(self.expert_samples_out)))
        # self.loss = -(self.expert_loss + self.policy_loss)
        self.expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.expert_samples_out,
                                                                   labels=tf.zeros_like(self.expert_samples_out))
        self.policy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.policy_samples_out,
                                                                   labels=tf.ones_like(self.policy_samples_out))
        self.expert_loss_avg = tf.reduce_mean(self.expert_loss)
        self.policy_loss_avg = tf.reduce_mean(self.policy_loss)
        self.loss = tf.reduce_mean(self.expert_loss) + tf.reduce_mean(self.policy_loss)

        # self.predictions = self.discriminate.outputs[0]
        # self.labels = tf.placeholder(tf.float32, shape=(None), name='y')
        # self.loss = tf.nn.sigmoid.cross_entropy_with_logits(self.out,self.labels)
        self.opt = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        # self.opt = tf.train.RMSPropOptimizer(self.alpha, decay=0.9).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        with open(myconfig['output_dir']+'/disc_train_loss_log.csv', 'w') as disc_train_log:
            disc_train_log.write("epoch,total_loss,expert_loss,policy_loss\n")


    def get_trainable_weights(self):
        return self.sess.run([self.discriminate.trainable_weights], feed_dict={})[0]

    def train(self, expert_samples_observations, expert_samples_actions, policy_samples_observations, policy_samples_actions):
        with open(myconfig['output_dir']+'/disc_train_loss_log.csv', 'a') as disc_train_log:
            loss_run = 0
            loss_before_train = 0
            for i in range(self.epochs):
                _, loss_run, expert_loss_run, policy_loss_run = self.sess.run([self.opt, self.loss, self.expert_loss_avg, self.policy_loss_avg],
                                            feed_dict={
                                                self.expert_samples_observations:
                                                    expert_samples_observations,
                                                self.expert_samples_actions:
                                                    expert_samples_actions,
                                                self.policy_samples_observations:
                                                    policy_samples_observations,
                                                self.policy_samples_actions:
                                                    policy_samples_actions
                                            })
                if i == 0:
                    loss_before_train = loss_run

                disc_train_log.write(str(i)+"," + str(loss_run)+"," + str(expert_loss_run)+"," + str(policy_loss_run)+"\n")
                # print('Discriminator loss:', loss_run)
                # if i % 100 == 0: print(i, "loss:", loss_run)
        return loss_before_train, loss_run

    def predict(self, samples_observations, samples_actions):
        return self.sess.run(self.log_D,
                             feed_dict={self.observations_input: samples_observations,
                                        self.actions_input: samples_actions})




class TRPOAgent(object):

    def __init__(self, env, observation_dimensions=11, latent_dimensions=5, action_dimensions=3):
        """
        Initializes the agent's parameters and constructs the flowgraph.
        :param env: Environment
        :param observation_dimensions: Observations' dimensions.
        :param action_dimensions: Actions' dimensions.
        """
        self.latent_list = []
        self.latent_sequence1 = []
        self.latent_sequence_prob = []
        self.encoder_rew = []
        self.encoder_rew_reset = []
        self.counter = 0
        self.counter2 = 0
        self.init = True
        self.env = env
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.latent_dimensions = latent_dimensions
        self.path_size = myconfig['path_size']
        self.mini_batch_size = myconfig['mini_batch_size']
        self.mini_batches = 10#myconfig['mini_batches'] #changed
        self.gamma = myconfig['gamma']
        self.lamda = myconfig['lamda']
        self.max_kl = myconfig['max_kl']
        self.total_episodes = 0
        self.logstd = np.float32(myconfig['logstd'])
        self.critic = Critic(observation_dimensions=self.observation_dimensions)
        self.discriminator = Discriminator(observation_dimensions=self.observation_dimensions, action_dimensions=self.action_dimensions)
        # self.replay_buffer = ReplayBuffer()
        self.sess = tf.Session(config=config)
        #self.enc_model = Clonos.create_Clonos()
        self.model2, self.encoder_x, self.l1, self.l2 = ClonosEncoder.create_encoder(self.observation_dimensions, self.latent_dimensions)
        self.model, self.x, self.h, self.h1 = Policy.create_policy(self.observation_dimensions, self.latent_dimensions, self.action_dimensions)

        visualize(self.model.trainable_weights)

        self.episode_history = deque(maxlen=100)

        self.advantages_ph = tf.placeholder(tf.float32, shape=None)
        self.actions_ph = tf.placeholder(tf.float32, shape=(None, action_dimensions),)
        self.old_log_prob_ph = tf.placeholder(tf.float32, shape=None)
        self.theta_ph = tf.placeholder(tf.float32, shape=None)
        self.tangent_ph = tf.placeholder(tf.float32, shape=None)
        self.mu_old_ph = tf.placeholder(tf.float32, shape=(None, action_dimensions))

        self.encoder_logits = self.model2.outputs[0]
        self.logits = self.model.outputs[0]

        #EDW
        self.q = self.encoder_logits
        self.argmax_q = tf.argmax(self.q, axis=1)

        #self.log_q = tf.log(self.argmax_q)
        var_list = self.model.trainable_weights
        self.flat_vars = get_flat(var_list)
        self.sff = set_from_flat(self.theta_ph, var_list)

        self.step_direction = tf.placeholder(tf.float32, shape=None)
        self.g_sum = gradient_summary(self.step_direction, var_list)

        # Compute surrogate.
        self.log_prob = gauss_log_prob(self.logits, self.logstd, self.actions_ph)
        neg_lh_divided = tf.exp(self.log_prob - self.old_log_prob_ph)
        w_neg_lh = neg_lh_divided * self.advantages_ph
        self.surrogate = tf.reduce_mean(w_neg_lh)

        kl_op = kl(self.logits, self.logstd, self.mu_old_ph, self.logstd)
        self.losses = [self.surrogate, kl_op]

        self.flat_grad = flat_gradient(self.surrogate, var_list)
        # Compute fisher vector product
        self_kl_op = self_kl(self.logits, self.logstd)
        self_kl_flat_grad = flat_gradient(self_kl_op, var_list)
        g_vector_dotproduct = tf.reduce_sum(self_kl_flat_grad * self.tangent_ph)
        # self.self_kl_grad = tf.gradients(self_kl_op, var_list)
        # start = 0
        # tangents = []
        # for var in var_list:
        #     end = start+np.prod(var.shape)
        #     tangents.append(tf.reshape(tangent_ph[start:end],var.shape))
        #     start = end
        # g_vector_product = [tf.reduce_sum(g * t) for (g, t) in zip(
        # self_kl_grad, tangents)]
        self.fvp = flat_gradient(g_vector_dotproduct, var_list)
        self.merged =  tf.compat.v1.summary.merge_all()
        self.train_writer =  tf.compat.v1.summary.FileWriter(myconfig['log_dir'], self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def predict(self, samples_observations):
        q_sess, q_arg = self.sess.run([self.q, self.argmax_q], feed_dict={self.encoder_x: samples_observations})
        #print('q: \n', q_sess)
        #print('q_argmax:\n', q_arg)
        #print('len_q:', len(q_sess), 'len_q_arg2:', len(q_arg))

        return q_sess, q_arg

    def __fisher_vector_product(self, g, feed):
        """
        Computes fisher vector product H*g using the direct method.
        :param p: Gradient of surrogate g.
        :param feed: Dictionary, feed_dict for tf.placeholders.
        :return: Fisher vector product H*g.
        """
        damping = myconfig['fvp_damping']
        feed[self.tangent_ph] = g
        fvp_run = self.sess.run(self.fvp, feed)
        assert fvp_run.shape == g.shape, "Different shapes. fvp vs g"
        return fvp_run + g * damping

    def get_vars(self):
        model2_weights = self.model2.weights
        model_weights = self.model.weights
        return model2_weights, model_weights

    def encoder_rewards(self, prob_latents):
        winner = np.argmax(prob_latents)
        encoder_reward = np.log(prob_latents[winner])
        #print('r:',encoder_reward)
        self.encoder_rew.append([encoder_reward])
        #array = np.asarray(self.encoder_rew[:])

    def one_hot(self, argmax):
        latent = []
        if(argmax==0):
            latent = [1, 0, 0]
            latent = np.asarray(latent)
        elif(argmax==1):
            latent = [0, 1, 0]
            latent = np.asarray(latent)
        elif(argmax==2):
            latent = [0, 0, 1]
            latent = np.asarray(latent)
        return latent

    def one_hot_encoding(self, x):
        #print('OneHotEncoding...')
        argmax = np.argmax(x)
        #print('MODE:', argmax)
        encoded = to_categorical(argmax, num_classes=5)
        #print('enc:', encoded)
        return encoded.tolist(), argmax

    def apply_oneHotEncoder(self, x):
        print('OneHotEncoding...')
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
        # columnTransformer = ColumnTransformer([('encoder', preprocessing.OneHotEncoder(), [0])], remainder='passthrough')
        # data_encoded = np.array(columnTransformer.fit_transform((df), dtype=np.str))
        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(x)
        print(integer_encoded)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        data_encoded = one_hot_encoder.fit_transform(integer_encoded)
        # invert first example
        inverted = label_encoder.inverse_transform([np.argmax(data_encoded[0, :])])
        print('inverted:', inverted)
        return data_encoded

    def keras_oneHotEncoder(self, x, shape):
        print('OneHotEncoding...')
        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(x)
        print(integer_encoded)
        encoded = to_categorical(integer_encoded, num_classes=shape)
        print('enc:', encoded)
        #print('test:',encoded[0])
        argmax = np.argmax(integer_encoded)
        print('argmax:', argmax)
        latent_new = encoded[argmax]
        print('new_latent:', latent_new)
        return latent_new

    def latent_sequence(self, prob_latents):
        winner = np.argmax(prob_latents)
        print(prob_latents)
        #winner = np.argwhere(prob_latents == np.amax(prob_latents))
        print('winner1:', winner)
        #if(winner==1):
        #    exit(0)
        #print(winner.flatten().tolist())  # if you want it as a list
        self.latent_list.append(prob_latents[winner])
        print(self.latent_list[:])
        array = np.asarray(self.latent_list[:])

        array_shape = array.shape[0]
        print('shape:', array_shape)

        #print(self.latent_list[:2])
        winner2 = np.argmax(self.latent_list[0:])
        print('winner2:', winner2)
        #exit(0)

        #argmax_mu2 = np.argmax(mu2)
        #latent = self.one_hot(argmax_mu2)
        return array, array_shape

    def act(self, observation, latent_seq):#, latent):
        global global_concat
        global obs_matrix

        decoder_input = np.concatenate((np.asarray(observation), latent_seq[self.counter]))
        obs_matrix = decoder_input
        mu = self.sess.run(self.logits, feed_dict={self.x: [decoder_input]})[0]

        act = mu + self.logstd * np.random.randn(self.action_dimensions)
        self.counter += 1

        return act, mu#, log_q #m2

    def init_encoder(self, observation):
        #init_latent = [1., 0., 0.]
        init_latent = [1., 0., 0., 0., 0.]
        init_latent = np.asarray(init_latent)
        enc_input = np.concatenate((np.asarray(observation), init_latent))
        # print('obs_l:', enc_input)
        mu2 = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input]})[0]
        latent_prob = np.asarray(mu2)
        latent, arg = self.one_hot_encoding(latent_prob)

        decoder_input = np.concatenate((np.asarray(observation), latent))
        return decoder_input, latent, arg

    def new_encoder(self, observation, latent_new):
        global global_concat_test

        if self.init == True:
            enc_input_n = np.concatenate((np.asarray(observation), latent_new))
            # print('obs_l:',enc_input_n)
        else:
            enc_input_n = np.concatenate((np.asarray(observation), global_concat_test))
            # print('obs_l:', enc_input_n)
        mu2 = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input_n]})[0]

        latent_prob = np.asarray(mu2)
        latent_new, arg = self.one_hot_encoding(latent_prob)

        # array, shape = self.latent_sequence(mu2)

        # one_hot_enc = self.keras_oneHotEncoder(array, shape)

        global_concat_test = latent_new
        decoder_input = np.concatenate((np.asarray(observation), latent_new))
        return decoder_input, latent_new, arg

    def act_test(self, decoder_input):#, latent):
        mu = self.sess.run(self.logits, feed_dict={self.x: [decoder_input]})[0]
        act = mu
        return act


    def run(self, episode_num, vae=False, bcloning=False, fname='0%_validate'):
        if vae:
            out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_vae_results.csv'
            action_out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_actions_vae_results.csv'

            enc_saver = tf.train.Saver(self.model2.weights)
            saver = tf.train.Saver(self.model.weights)

            enc_saver.restore(self.sess,"../VAE/results_gumbel_softmax/checkpoint/run9(5modes)/trpo_plugins/encoder/encoder_model_e2000.ckpt")
            saver.restore(self.sess,"../VAE/results_gumbel_softmax/checkpoint/run9(5modes)/trpo_plugins/decoder/decoder_model_e2000.ckpt")

            with open(out_file, 'w') as results, open(action_out_file, 'w') as action_results:
                # results.write("episode,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11\n")
                results.write(
                    "episode,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,latent1,latent2,latent3,latent4,latent5\n")
                action_results.write("episode,action1,action2,action3\n")
                actions = []
                obs = []
                discounted_rewards = []
                total_rewards = []
                print('Episodes,Reward')

                for i_episode in range(episode_num):
                    self.latent_list = []
                    self.counter2 = 0
                    self.init = True
                    steps = 0
                    r_i = []
                    observation = self.env.reset()
                    line = ""
                    # EDW ALLAXTHKE
                    dec_input1, latent_init, _ = self.init_encoder(observation)
                    # print('observation:', observation)
                    # print('INIT_OBSERVATION:', dec_input1)

                    for ob in dec_input1:
                        line += "," + str(ob)
                    results.write(str(i_episode) + line + "\n")

                    # action_results.write(
                    #     str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," +
                    #     str(action[2]) + "\n")
                    total_reward = 0

                    for t in range(self.path_size):
                        # print('COUNTER2:', self.counter2)

                        if self.init:
                            obs.append(dec_input1)
                            action = self.act_test(dec_input1)
                            action = action.tolist()
                            # print('action:', action, '\n', 'late:', latent_n, 'dec:', dec_input)
                        else:
                            obs.append(dec_input2)
                            action = self.act_test(dec_input2)
                            action = action.tolist()

                        # print('EDW:', action[0], action[1], action[2])
                        self.counter2 += 1
                        self.env.render()
                        observation2, reward, done, _ = self.env.step(action)
                        dec_input2, latent,_ = self.new_encoder(observation2, latent_init)
                        # print('observation2:', observation2)
                        # print('_OBSERVATION2:', dec_input2)
                        steps += 1
                        r_i.append(reward)
                        actions.append(action)
                        total_reward += reward
                        # action = unnormalize_action(action)

                        if not done:
                            line = ""
                            for ob in dec_input2:
                                line += "," + str(ob)
                            results.write(str(i_episode) + line + "\n")

                        action_results.write(
                            str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," + str(action[2]) + "\n")

                        self.init = False
                        if t % 100 == 0:
                            print("%i/%i" % (t + 100, self.path_size))
                        if t >= self.path_size or done:
                            # if done:
                            #    print('')
                            # exit(0)
                            # continue
                            break

                    print('{0},{1}'.format(i_episode, total_reward))
                    # exit(0)
                    discounted_rewards.extend(discount(r_i, self.gamma))
                    total_rewards.append(total_reward)

        elif bcloning:
            out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_bcloning_results.csv'
            action_out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_actions_bcloning_results.csv'

            saver = tf.train.Saver(self.model.weights)
            saver.restore(self.sess, "./checkpoint/bcloning.ckpt")
        else:
            out_file = myconfig['output_dir']+'/exp'+str(myconfig['exp'])+'_'+fname+'_D-Info_GAIL_results.csv'
            action_out_file = myconfig['output_dir'] + '/exp'+str(myconfig['exp'])+'_'+fname+'_D-Info_GAIL_actions_results.csv'

            enc_saver = tf.train.Saver(self.model2.weights)
            saver = tf.train.Saver(self.model.weights)
            enc_saver.restore(self.sess,"../VAE/results_gumbel_softmax/checkpoint/run9(5modes)/trpo_plugins/encoder/encoder_model_e2000.ckpt")
            saver.restore(self.sess, "./.output/exp" + str(myconfig['exp']) + "model.ckpt")

            with open(out_file, 'w') as results, open(action_out_file, 'w') as action_results:
                #results.write("episode,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11\n")
                results.write("episode,var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,latent1,latent2,latent3,latent4,latent5\n")
                action_results.write("episode,action1,action2,action3\n")
                actions = []
                obs = []
                discounted_rewards = []
                total_rewards = []
                print('Episodes,Reward')
                for i_episode in range(episode_num):
                    self.latent_list = []
                    self.counter2 = 0
                    self.init = True
                    steps = 0
                    r_i = []
                    observation = self.env.reset()
                    line = ""
                    # EDW ALLAXTHKE
                    dec_input1, latent_init, _ = self.init_encoder(observation)
                    # print('observation:', observation)
                    # print('INIT_OBSERVATION:', dec_input1)

                    for ob in dec_input1:
                        line += "," + str(ob)
                    results.write(str(i_episode) + line + "\n")

                    # action_results.write(
                    #     str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," +
                    #     str(action[2]) + "\n")
                    total_reward = 0

                    for t in range(self.path_size):
                        # print('COUNTER2:', self.counter2)

                        if self.init:
                            obs.append(dec_input1)
                            action = self.act_test(dec_input1)
                            action = action.tolist()
                            # print('action:', action, '\n', 'late:', latent_n, 'dec:', dec_input)
                        else:
                            obs.append(dec_input2)
                            action = self.act_test(dec_input2)
                            action = action.tolist()

                        # print('EDW:', action[0], action[1], action[2])
                        self.counter2 += 1
                        self.env.render()
                        observation2, reward, done, _ = self.env.step(action)
                        dec_input2, latent, _ = self.new_encoder(observation2, latent_init)
                        # print('observation2:', observation2)
                        # print('_OBSERVATION2:', dec_input2)
                        steps += 1
                        r_i.append(reward)
                        actions.append(action)
                        total_reward += reward
                        # action = unnormalize_action(action)

                        if not done:
                            line = ""
                            for ob in dec_input2:
                                line += "," + str(ob)
                            results.write(str(i_episode) + line + "\n")

                        action_results.write(
                            str(i_episode) + "," + str(action[0]) + "," + str(action[1]) + "," + str(action[2]) + "\n")

                        self.init = False
                        if t % 100 == 0:
                            print("%i/%i" % (t + 100, self.path_size))
                        if t >= self.path_size or done:
                            # if done:
                            #    print('')
                            # exit(0)
                            # continue
                            break

                    print('{0},{1}'.format(i_episode, total_reward))
                    # exit(0)
                    discounted_rewards.extend(discount(r_i, self.gamma))
                    total_rewards.append(total_reward)
        return actions, obs, discounted_rewards, total_rewards

    def rollout(self, mini_batch, latent_sequence_np):
        not_enough_samples = True
        batch_actions = []
        batch_observations = []
        batch_observations_lat = []
        batch_total_env_rewards = []
        log_observations = []
        log_actions = []
        episode = 0
        samples = 0
        index=0
        index2=1000

        while not_enough_samples:
            episode += 1
            self.total_episodes += 1
            actions = []
            observations = []
            observations_lat = []
            observation = self.env.reset()
            #print('obs:',observation)
            total_env_reward = 0
            latent_sequence = latent_sequence_np[index:index2]
            #print('len:', len(latent_sequence), 'seq:', latent_sequence)
            #print(latent_sequence[250], ' ')

            for t in range(self.path_size):

                #observation = np.concatenate((observation, latents[t]))
                #action = self.act(observation, latents[t])[0].tolist()

                #EDW ALLAXTHKE
                #action = self.act(observation)[0].tolist()
                action, _ = self.act(observation, latent_sequence)
                action = action.tolist()
                #print('action:', action)
                observation_lat = obs_matrix.tolist()
                #print('obs:', observation)
                #print('obs_lat:', observation_lat)
                observations.append(observation)
                observations_lat.append(observation_lat)
                #exit(0)
                if mini_batch % 100 == 0:
                    # Print global observation
                    #print('EDW:', observation_lat)
                    #EDW ALLAXTHKE
                    obs_ep = observation_lat + [episode]
                    #obs_ep = np.concatenate((observation, [episode]))
                    #print('obs_ep:', obs_ep)
                    obs_ep = np.asarray(obs_ep)
                    # print('obs_ep:', obs_ep)
                    log_observations.append(obs_ep)
                    # print('log_observs:', log_observations, '\n')
                    log_action = copy.deepcopy(action)
                    log_action = np.append(log_action, episode)
                    log_actions.append(log_action)

                self.env.render()
                observation, reward_env, done, _ = self.env.step(action)
                #print(observation)
                #print(reward_env)
                #print(done)
                total_env_reward += reward_env

                actions.append(action)

                if t % 100 == 0:
                    #print("%i/%i" % (t + 100, self.path_size))
                    continue
                if done:
                    #print('self.counter:', self.counter)
                    index += 1000
                    index2 += 1000
                    self.counter = 0
                    if(index2>=49999):
                        index=0
                        index2=1000

                    break
                    #continue

            samples += len(actions)
            batch_observations_lat.append(observations_lat)
            batch_observations.append(observations)
            batch_actions.append(actions)
            batch_total_env_rewards.append(total_env_reward)
            if samples >= self.mini_batch_size:
                not_enough_samples = False
                self.counter = 0
        if mini_batch % 100 == 0:
            np.savetxt(str(myconfig['exp'])+'_'+str(mini_batch)+"_observation_log.csv", np.asarray(log_observations)
                       , delimiter=',', header='var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,latent1,latent2,latent3,latent4,latent5,episode',
                       comments='')
            np.savetxt(str(myconfig['exp'])+'_'+str(mini_batch) + "_action_log.csv", np.asarray(log_actions),
                       delimiter=',', header='action1,action2,action3,' + 'episode', comments='')
        print('Hello from rollout... this is batch total env rewards...', len(batch_total_env_rewards))
        return batch_observations_lat, batch_observations, batch_actions, batch_total_env_rewards


    def run_clonos(self, observation, init_latent):
        print('Running.. Encoder Clone')
        global global_concat_test1
        state = 0
        latent_counter = 1
        latent_flag = True
        init_latent = np.asarray(init_latent)
        while state < 50016:
            #print('state:', state)
            #if (latent_counter % 1000 == 0 or latent_counter == 1):
            if (latent_counter == 1):
                latent_flag = True
            else:
                latent_flag = False
                # print('counter:', counter, 'flag:', latent_flag)

            if (latent_flag == True):
                enc_input = np.concatenate((np.asarray(observation[state]), init_latent))
            else:
                enc_input = np.concatenate((np.asarray(observation[state]), global_concat_test1))

            latent_prob = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input]})[0]
            self.latent_sequence_prob.append(latent_prob)
            latent_prob = np.asarray(latent_prob)
            latent_new, argmax = self.one_hot_encoding(latent_prob)
            self.latent_sequence1.append(latent_new)
            sequence_prob = self.latent_sequence_prob
            sequence = self.latent_sequence1
            #print(latent_new)
            #print(sequence_prob[state])
            #print(sequence[state])
            # print(sequence)
            global_concat_test1 = latent_new
            state += 1
            latent_counter += 1
        self.latent_sequence_list = copy.deepcopy(self.latent_sequence1)
        np_arr_prob = np.asarray(self.latent_sequence_prob)
        np_arr = np.asarray(self.latent_sequence1)
        print(np_arr_prob.shape)
        print(np_arr.shape)
        print('Encoder Clone finished, mode sequences were created successfully!')
        #latent_sequence_prob_pd = pd.DataFrame(self.latent_sequence_prob, columns=['latent1', 'latent2', 'latent3'])
        #latent_sequence_pd = pd.DataFrame(self.latent_sequence1, columns=['latent1', 'latent2', 'latent3'])
        #latent_sequence_prob_pd.to_csv('./expert_data/latent_sequence_prob.csv', index=False)
        #latent_sequence_pd.to_csv('./expert_data/latent_sequence.csv', index=False)

        return np_arr_prob, np_arr

    def train(self, expert_observations, expert_actions):
        """
        Trains the agent.
        :return: void
        """

        # self.replay_buffer.seed_buffer(expert_observations, expert_actions)
        encoder_saver = tf.train.Saver(self.model2.weights)
        encoder_saver.restore(self.sess, "../VAE/results_gumbel_softmax/checkpoint/run9(5modes)/trpo_plugins/encoder/encoder_model_e2000.ckpt")
        saver = tf.train.Saver(self.model.weights)
        #saver.restore(self.sess, "./checkpoint/bcloning.ckpt")
        saver.restore(self.sess, "../VAE/results_gumbel_softmax/checkpoint/run9(5modes)/trpo_plugins/decoder/decoder_model_e2000.ckpt")
        discriminator_saver = tf.train.Saver(self.discriminator.discriminate.weights)

        latent_sequence_prob_np, latent_sequence_np = self.run_clonos(expert_observations, init_latent=[1., 0., 0., 0., 0.])
        #print(latent_sequence_prob_np[:100], latent_sequence_prob_np.shape)
        #print(latent_sequence_np[:100], latent_sequence_np.shape)


        print('Batches,Episodes,Surrogate,Reward,Env Reward')
        my_env_rew=[]
        for mini_batch in range(self.mini_batches+1):
            # expert_observations_batch, expert_actions_batch = self.replay_buffer.get_batch(self.mini_batch_size)
            expert_observations_batch = expert_observations
            expert_actions_batch = expert_actions

            batch_observations_lat, batch_observations, batch_actions, batch_total_env_rewards = self.rollout(mini_batch, latent_sequence_np)


            flat_actions = [a for actions in batch_actions for a in actions]
            flat_observations = [o for observations in batch_observations_lat for o in observations]

            flat_actions = np.asarray(flat_actions, dtype=np.float32)
            flat_observations = np.asarray(flat_observations, dtype=np.float32)
            print('len:', len(flat_observations))
            flat_observations2 = flat_observations[:, :11]

            if mini_batch < self.mini_batches:
                d_loss_before_train, discriminator_loss = self.discriminator.train(expert_observations_batch, expert_actions_batch,
                                         flat_observations2[:self.mini_batch_size],
                                         flat_actions[:self.mini_batch_size])
            else:
                print('discriminator train not')

            batch_total_rewards = []
            batch_discounted_rewards_to_go = []
            batch_advantages = []
            total_reward = 0

            global d
            d = 0
            counters = 0
            index_rew = 0
            index_rew2 = 0
            for (observations, actions, obs_lat) in zip(batch_observations, batch_actions, batch_observations_lat):
                counters += len(observations)
                rewards_q, argmax_q = self.predict(np.asarray(obs_lat))
                #print('len:', len(rewards_q))
                #argmax_q = [np.argmax(i) for i in rewards_q]
                rewards_q = np.asarray(rewards_q)
                #print('rewards_q: \n', rewards_q)
                #print('argmax:', argmax_q)
                #print('len_argmax:', len(argmax_q))

                argmax_q2 = []
                for reward_q, i in zip(rewards_q, argmax_q):
                    element = reward_q[i]
                    argmax_q2.append(element)

                #print('argmax2:', argmax_q2)
                #print('len_argmax2:', len(argmax_q2))

                rewards_log = [np.log(i) for i in argmax_q2]
                #print('rewards_log:', rewards_log)
                #print('len_rew_log:', len(rewards_log))

                #rewards_q2 = [i * 0.01 for i in rewards_log]
                rewards_q2 = np.asarray([[i * 0.01] for i in rewards_log])
                #print('rewards_q:', rewards_q2)
                #print('len_rew_q:', len(rewards_q2))

                reward_t = -self.discriminator.predict(np.array(observations), np.array(actions))
                #print('rewards_t:', reward_t)
                #print('len_rew_t:', len(reward_t))
                #reward_t = [[i+t for i,t in zip(y, e)] for (y, e) in zip(reward_t, rewards_q2)]
                #print('reward_t_len:', len(reward_t), 'reward_q_len:', len(rewards_q2))
                #reward_t = reward_t + rewards_q2
                #print('reward_t_len:', len(reward_t))
                reward_t = np.asarray([i+j for i,j in zip(reward_t, rewards_q2)])
                #reward_t = [[sum(i) for i in y] for y in zip(reward_t, rewards_q2)]

                #print('rewards_t-after:', reward_t, '\n, type:', type(reward_t))

                total_reward += np.sum(reward_t)

                #print('\n Total_Rew:', total_reward)
                batch_total_rewards.append(total_reward)
                reward_t = (reward_t.flatten())#.tolist()

                batch_discounted_rewards_to_go.extend(discount(reward_t, self.gamma))
                obs_episode_np = np.array(observations)
                v = np.array(self.critic.predict(obs_episode_np)).flatten()
                #print('v:', v)
                v_next = shift(v, -1, cval=0)
                #print('v_next:', v_next)
                undiscounted_advantages = reward_t + self.gamma * v_next - v
                #print('undisc_adv:', undiscounted_advantages)
                #print('len_und:', len(undiscounted_advantages))

                discounted_advantages = discount(undiscounted_advantages, self.gamma * self.lamda)
                #print('disc_adv:', discounted_advantages)
                batch_advantages.extend(discounted_advantages)

                d+=1
            #print('d:', d)
            #print('counters:', counters)

            #print('\n')

            discounted_rewards_to_go_np = np.array(batch_discounted_rewards_to_go)
            discounted_rewards_to_go_np.resize((self.mini_batch_size, 1))

            observations_np = np.array(flat_observations2, dtype=np.float32) #11
            #print('obs_np:', observations_np[0])
            observations_np2 = np.array(flat_observations, dtype=np.float32) #14
            #print('obs_np2:', observations_np2[0])
            observations_np.resize((self.mini_batch_size, self.observation_dimensions))
            observations_np2.resize((self.mini_batch_size, self.observation_dimensions+self.latent_dimensions))

            advantages_np = np.array(batch_advantages)
            advantages_np.resize((self.mini_batch_size,))

            actions_np = np.array(flat_actions, dtype=np.float32).flatten()
            actions_np.resize((self.mini_batch_size, self.action_dimensions))

            self.critic.train(observations_np, discounted_rewards_to_go_np)
            feed = {self.x: observations_np2,
                    self.actions_ph: actions_np,
                    self.advantages_ph: advantages_np,
                    self.old_log_prob_ph: self.sess.run([self.log_prob], feed_dict={self.x: observations_np2, self.actions_ph: actions_np})
                    }

            g = np.array(self.sess.run([self.flat_grad], feed_dict=feed)[0], dtype=np.float32)
            step_dir = conjugate_gradient(self.__fisher_vector_product, g, feed)
            fvp = self.__fisher_vector_product(step_dir, feed)
            shs = step_dir.dot(fvp)
            assert shs > 0
            fullstep = np.sqrt(2 * self.max_kl / shs) * step_dir

            def loss_f(theta, mu_old):
                """
                Computes surrogate and KL of weights theta, used in
                line search.
                :param theta: Weights.
                :param mu_old: Distribution of old weights.
                :return: Vector [surrogate,KL]
                """
                feed[self.theta_ph] = theta
                feed[self.mu_old_ph] = mu_old
                self.sess.run([self.sff], feed_dict=feed)
                return self.sess.run(self.losses, feed_dict=feed)

            surrogate_run = self.sess.run(self.surrogate, feed_dict=feed)

            mu_old_run = self.sess.run(self.logits, feed_dict={self.x: observations_np2})
            theta_run = np.array(self.sess.run([self.flat_vars], feed_dict={})[0], dtype=np.float32)

            theta_new, surrogate_run = line_search(loss_f, theta_run,
                                                   fullstep, mu_old_run,
                                                   g.dot(step_dir),
                                                   surrogate_run,
                                                   self.max_kl)

            feed[self.theta_ph] = theta_new
            feed[self.step_direction] = step_dir
            print("self.stepdirc....",self.step_direction)
            print("self.stepmerged....",self.merged)

            tf.summary.scalar('minibatch', mini_batch)

            self.merged =  tf.compat.v1.summary.merge_all()
            print("self.stepmerged....",self.merged)
            _ = self.sess.run([self.sff], feed_dict=feed)
            if (mini_batch % 10 == 0):
                _, summary = self.sess.run([self.step_direction, self.merged], feed_dict=feed)
                #summary = self.sess.run([self.step_direction, self.merged], feed_dict=feed)

                self.train_writer.add_summary(summary, mini_batch)

            self.episode_history.append(np.mean(batch_total_env_rewards))
            # mean = np.mean(self.episode_history)
            # if mean > max_mean:
            #     max_mean = mean
            #     saver.save(self.sess, myconfig['output_dir']+"/exp"+str(myconfig['exp'])+"model.ckpt")

            print('{0},{1},{2},{3},{4},{5},{6}'.format(mini_batch, self.total_episodes,
                                           surrogate_run,np.mean(batch_total_rewards),
                                           np.mean(batch_total_env_rewards), d_loss_before_train, discriminator_loss))
            my_env_rew.append(np.mean(batch_total_env_rewards))
            if mini_batch % 100 == 0:
                #encoder_saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "encoder_model.ckpt", global_step=mini_batch)
                saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "model.ckpt",global_step=mini_batch)
                discriminator_saver.save(self.discriminator.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "discriminator.ckpt", global_step=mini_batch)

        #encoder_saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "encoder_model.ckpt")
        saver.save(self.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "model.ckpt")

        import matplotlib.pyplot  as plt
        #print("before plotting... total env rewards is. -------------- my_env_rew-------------")
        #print(batch_total_env_rewards)
        plt.plot(my_env_rew)
        # plt.show()

        plt.savefig('foo.png')
        discriminator_saver.save(self.discriminator.sess, myconfig['output_dir'] + "output/exp" + str(myconfig['exp']) + "discriminator.ckpt")