# from __future__ import print_function
# matplotlib.use('Agg')
import os
os.add_dll_directory("C:\\Users\\ETAPTA\\PycharmProjects\\HopperProject\\venv\\Lib\\site-packages\\mujoco_py\\binaries\\windows\\mujoco210\\bin")

import mujoco_py
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Input, Dense, concatenate, LeakyReLU, InputLayer, Softmax
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
# import tensorflow.contrib.slim as slim #works with 1.14 tensorflow
# import tensorflow_probability as tf_prob
# import tensorflow_datasets as tfds
from sklearn import preprocessing
from sklearn.utils import shuffle as shuffle1
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import os
import time
# from tensorflow.examples.tutorials.mnist import input_data
import pickle
from plot_utils import *
from matplotlib import pyplot as plt
import gym
from gym import wrappers
from utils import conjugate_gradient, set_from_flat, kl, self_kl, \
    flat_gradient, get_flat, discount, line_search, gauss_log_prob, visualize, gradient_summary, \
    unnormalize_action
tf.compat.v1.disable_eager_execution()
# sns.set_style('whitegrid')
sns.set()
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'Blues'
print("tf version")
print(tf. __version__)
# Define the different distributions
#distributions = tf.compat.v1.estimator.distributions
#bernoulli = distributions.Bernoulli
# distributions = tf_prob.distributions
# bernoulli = distributions.Bernoulli

# Define current_time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
# GPU Usage

config = tf.compat.v1.ConfigProto()  # (allow_soft_placement=False)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7


# Define Directory Parameters
flags = tf.compat.v1.app.flags
flags.DEFINE_string('data_dir', os.getcwd() + '\\dataset\\', 'Directory for data')
flags.DEFINE_string('results_dir', os.getcwd() + '.\\results_gumbel_softmax\\', 'Directory for results')
flags.DEFINE_string('checkpoint_dir', os.getcwd() + '\\results_gumbel_softmax\\checkpoint\\' + 'run9(5modes)',
                    'Directory for checkpoints')  # current_time


flags.DEFINE_string('models_dir', os.getcwd() + '\\results_gumbel_softmax\\model\\', 'Directory for models')

# Define Model Parameters
flags.DEFINE_integer('batch_size', 32, 'Minibatch size')
flags.DEFINE_integer('num_iters', 50000, 'Number of iterations')
flags.DEFINE_integer('epochs', 200, 'Number of epochs')#changed
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')  # 0.0001
# flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
# flags.DEFINE_integer('num_cat_dists', 200, 'Number of categorical distributions') # num_cat_dists//num_calsses
flags.DEFINE_integer('num_actions', 3, 'Number of classes')
flags.DEFINE_float('init_temp', 5.0, 'Initial temperature')
flags.DEFINE_float('min_temp', 0.1, 'Minimum temperature')  # 0.5
flags.DEFINE_float('anneal_rate', 0.00003, 'Anneal rate')
flags.DEFINE_bool('straight_through', False, 'Straight-through Gumbel-Softmax')
flags.DEFINE_string('kl_type', 'relaxed', 'Kullback-Leibler divergence (relaxed or categorical)')
flags.DEFINE_bool('learn_temp', False, 'Learn temperature parameter')

flags.DEFINE_integer('encoder_n_input', 11, 'input size')
flags.DEFINE_integer('n_hidden_encoder_layer_1', 100, 'encoder_layer1 size')
flags.DEFINE_integer('n_hidden_encoder_layer_2', 100, 'encoder_layer2 size')
flags.DEFINE_integer('latent_nz', 5, 'latent size')
flags.DEFINE_integer('decoder_n_input', 11, 'input size')
flags.DEFINE_integer('n_hidden_decoder_layer_1', 100, 'decoder_layer1 size')
flags.DEFINE_integer('n_hidden_decoder_layer_2', 100, 'decoder_layer2 size')
FLAGS = flags.FLAGS


# WEIGHTS INITIALIZATION
def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def glorot_init(shape):
    return tf.random.normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# PREPROCESSING

obs = pd.read_csv('..\\dataset\\hopper-v2\\hopper_v2_observations(50016).csv')
obs_test = pd.read_csv('..\\dataset\\hopper-v2\\hopper_v2_observations(62000).csv')
obs_train = np.asarray(obs)
obs_test = np.asarray(obs_test)
print('OBS:', obs_train.shape)
obs_test = obs_test[:50016]
print('OBS_TEST:', obs_test.shape)
actions = pd.read_csv('..\\dataset\\hopper-v2\\hopper_v2_actions(50016).csv')
actions_test = pd.read_csv('..\\dataset\\hopper-v2\\hopper_v2_actions(62000).csv')
actions_train = np.asarray(actions)
actions_test = np.asarray(actions_test)
actions_test = actions_test[:50016]

# obs = np.array(obs).
# actions = np.array(actions)

print("Observations: \n", obs)
print("Actions: \n", actions)


# Normalize
def apply_normalization_observations(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm_obs = pd.DataFrame(np_scaled,
                               columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10',
                                        'var11'])
    return df_norm_obs


def apply_normalization_actions(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm = pd.DataFrame(np_scaled, columns=['action1', 'action2', 'action3'])
    return df_norm


# OneHotEncoder
def apply_oneHotEncoder(X, y):
    print('OneHotEncoding...')
    one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    # columnTransformer = ColumnTransformer([('encoder', preprocessing.OneHotEncoder(), [0])], remainder='passthrough')
    # data_encoded = np.array(columnTransformer.fit_transform((df), dtype=np.str))
    data_encoded = one_hot_encoder.fit_transform(X, y)
    data_encoded = np.array(data_encoded)
    return data_encoded


def pandas1(obs):
    df_norm_obs = pd.DataFrame(obs,
                               columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10',
                                        'var11'])
    return df_norm_obs


def pandas_latent1(obs):
    df_norm_obs = pd.DataFrame(obs,
                               columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10',
                                        'var11', 'latent1', 'latent2', 'latent3'])
    return df_norm_obs


def pandas2(actions):
    df_norm = pd.DataFrame(actions, columns=['action1', 'action2', 'action3'])
    return df_norm


# Generate data from latent space
def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(input=logits))
    # y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    y = tf.keras.activations.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(input=logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(input_tensor=y, axis=1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


def encoder(x, latent, temperature):
    # Variational posterior q(y|x), i.e. the encoder (shape=(batch_size, 80))
    i = 0

    h1 = Dense(FLAGS.n_hidden_encoder_layer_1, activation='tanh', kernel_initializer='glorot_uniform',
               bias_initializer='zeros', name='encoder_h1')
    h2 = Dense(FLAGS.n_hidden_encoder_layer_2, activation='tanh', kernel_initializer='glorot_uniform',
               bias_initializer='zeros', name='encoder_h2')
    out = Dense(FLAGS.latent_nz, name='encoder_out')
    # enc = tf.zeros([1,14], tf.float32)
    # enc = tf.Variable([[1.21717712e+00, -1.07320348e-04, -4.18256945e-03, -3.89150316e-03,
    #   -2.80910121e-03,  3.92307339e-03,  2.90272282e-03, -4.71132621e-03,
    #   -3.49365975e-04,  5.18269952e-05,  4.63955574e-03,  1.,
    #    0.,  0.]], dtype=tf.float32)

    q_y_list = []
    log_q_y_list = []
    out_layer_list = []
    enc_inplayer_list = []
    while (i < 32):
        print('Iteration i:', i)
        feature = tf.gather(x, [i])
        print("Feature: ",feature)
        enc_inp = tf.concat([feature, latent], 1)
        print("enc_input layer: ",enc_inp)
        e = enc_inp
        # enc = tf.concat([enc, enc_inp], 0)
        # TENSORFLOW Keras Layers
        encoder_input_layer = Input(shape=(16,), batch_size=1, tensor=enc_inp, dtype=tf.float32)
        enc_inplayer_list.append(encoder_input_layer)
        _h1 = h1(encoder_input_layer)
        _h2 = h2(_h1)
        _out = out(_h2)  # , activation='softmax'
        out_layer_list.append(_out)
        #print('OUT:', out)
        # q_y = Softmax(out)
        q_y = tf.keras.activations.softmax(_out)
        q_y_temp = q_y
        q_y_list.append(q_y)

        log_q_y = tf.math.log(q_y + 1e-20)
        log_q_y_temp = log_q_y
        log_q_y_list.append(log_q_y)
        z = tf.reshape(gumbel_softmax(_out, temperature, hard=True), [-1, FLAGS.latent_nz])
        latent = z
        if (i < 1):
            # q_y_list = tf.concat([q_y_temp, q_y], axis=0)
            # log_q_y_list = tf.concat([log_q_y_temp, log_q_y], axis=0)

            latents = tf.concat([latent, z], axis=0)  # tf.Variable(z)
            enc = tf.concat([e, enc_inp], 0)
        else:
            # q_y_list = tf.concat([q_y_list, q_y], axis=0)
            # log_q_y_list = tf.concat([log_q_y_list, log_q_y], axis=0)

            latents = tf.concat([latents, z], axis=0)
            enc = tf.concat([enc, enc_inp], 0)
        print("latents: ", latents)
        print("enc: ", enc)

        i += 1
    # Encoder Model
    encoder_model = Model(inputs=enc_inplayer_list, outputs=out_layer_list)

    enc = tf.unstack(enc, 33)
    latents = tf.unstack(latents, 33)
    # q_y_list = tf.unstack(q_y_list, 33)
    # log_q_y_list = tf.unstack(log_q_y_list, 33)
    del enc[0]
    del latents[0]
    # del q_y_list[0]
    # del log_q_y_list[0]
    latents = latents
    enc = enc
    # q_y_list = q_y_list
    # log_q_y_list = log_q_y_list
    latents_shape = tf.shape(input=latents)
    enc_shape = tf.shape(input=enc)

    print('q_y_list:', q_y_list)
    print('log_q_y_list:', log_q_y_list)
    print('len1:', len(q_y_list))
    print('len2:', len(log_q_y_list))

    return out_layer_list, q_y_list, log_q_y_list, encoder_model, enc, latents, latents_shape, enc_shape


def decoder(x, latents):
    # Generative model p(x|y), i.e. the decoder (shape=(batch_size, 200))

    # y = tf.reshape(gumbel_softmax(logits_y, temperature, hard=False), [-1, FLAGS.num_cat_dists, FLAGS.num_classes])
    # z = tf.reshape(gumbel_softmax(out, temperature, hard=True), [-1, FLAGS.latent_nz])

    dec_inp = tf.concat([x, latents], 1)

    # Tensorflow Keras Layers
    decoder_input_layer = Input(shape=(16,), batch_size=32, tensor=dec_inp, dtype=tf.float32)
    dec_h1 = Dense(FLAGS.n_hidden_decoder_layer_1, activation='tanh', kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', name='decoder_h1')(
        decoder_input_layer)  # activation=tf.keras.layers.LeakyReLU(alpha=0.01), glorot_normal
    dec_h2 = Dense(FLAGS.n_hidden_decoder_layer_2, activation='tanh', kernel_initializer='glorot_uniform',
                   bias_initializer='zeros', name='decoder_h2')(dec_h1)
    # decoder_out = Dense(FLAGS.num_actions, name='decoder_out')(dec_h2)
    decoder_out = Dense(FLAGS.num_actions, name='decoder_out')(dec_h2)
    # decoder_model = Model(inputs=logits_y, outputs=decoder_out)
    softmax = tf.keras.activations.softmax(decoder_out)

    # decoder_out = tf.layers.flatten(decoder_out)
    # p_x = bernoulli(logits=softmax)#logits_x

    # Decoder Model
    decoder_model = Model(inputs=decoder_input_layer, outputs=decoder_out)

    return decoder_out, softmax, decoder_model, dec_inp


def create_train_op(y, lr, predicted):
    # NA TO DW
    loss = tf.reduce_mean(input_tensor=tf.square(predicted - y))
    # elbo = tf.reduce_sum(tf.square(predicted - y))
    # loss = tf.reduce_mean(elbo)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return train_op, loss


def create_train_op_kl(y, predicted, lr, q_y, log_q_y):
    # kl_tmp = tf.reshape(q_y * (log_q_y - tf.log(1.0 / FLAGS.num_classes)),[-1, FLAGS.num_cat_dists, FLAGS.num_classes])
    # KL = tf.reduce_sum(kl_tmp, [1,2])
    tmp1 = tf.math.log(1.0 / FLAGS.num_actions)
    tmp = tf.cast(tmp1, tf.float32)
    kl_tmp = tf.reshape(q_y * (log_q_y - tmp), [-1, FLAGS.num_actions])
    # KL = tf.reduce_sum(kl_tmp, [1,1])
    KL = tf.reduce_sum(input_tensor=kl_tmp, axis=1)

    # WITH KL DIVERGENCE
    # elbo = tf.reduce_sum(p_x.log_prob(x), 1) - KL
    elbo = tf.reduce_sum(input_tensor=tf.square(predicted - y)) - KL
    loss = tf.reduce_mean(input_tensor=-elbo)
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return train_op, loss


class Dataset:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def count_train_num_examples(self):
        return len(self.x_train), len(self.y_train)

    def count_test_num_examples(self):
        return len(self.x_test), len(self.y_test)

    def train_next_batch(self, counter):
        batch = self.x_train[counter:counter + 32]
        batch_y = self.y_train[counter:counter + 32]
        return batch, batch_y

    def test_next_batch(self, counter):
        batch = self.x_test[counter:counter + 32]
        batch_y = self.y_test[counter:counter + 32]
        return batch, batch_y


##########################################################################################

x_train = obs_train
x_test = obs_test
y_train = actions_train
y_test = actions_test

# TensorFlow Dataset
datas = Dataset(x_train, x_test, y_train, y_test)


def train():
    # Setup Encoder

    temperature = tf.compat.v1.placeholder(tf.float32, [], name='temperature')  # tf.float32 !!!...
    learning_rate = tf.compat.v1.placeholder(tf.float32, [], name='lr_value')
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, 11], name='enc_inp')
    y_holder = tf.compat.v1.placeholder(tf.float32, shape=[None, 3], name='y_holder')
    latent = tf.compat.v1.placeholder(tf.float32, shape=[None, 5], name='latent')

    initial_latent = [1., 0., 0., 0., 0.]
    initial_latent = np.asarray(initial_latent)
    initial_latent = np.expand_dims(initial_latent, axis=0)


    logits_y, q_y, log_q_y, encod_model, enc_inpp, latents, latents_shape, encc_shape = encoder(inputs, latent,
                                                                                                temperature)

    # Setup Decoder
    decoder_last_layer, softmax, dec_model, new_inp = decoder(inputs, latents)

    # Operations
    train_op, loss = create_train_op(y_holder, learning_rate, decoder_last_layer)  # NO_KL
    # train_op, loss = create_train_op_kl(y_holder, decoder_last_layer, learning_rate, q_y, log_q_y) # WITH KL

    # Initialize Session
    init_op = [tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()]
    sess = tf.compat.v1.Session(config=config)
    saver = tf.compat.v1.train.Saver(encod_model.weights)  # TI KANOUME SAVE
    # saver.restore(sess, "./results_gumbel_softmax/checkpoint/run5/encoder/encoder2000/encoder_model_e1999-i49999.ckpt")
    dec_saver = tf.compat.v1.train.Saver(dec_model.weights)  # TI KANOUME SAVE
    # dec_saver.restore(sess, "./results_gumbel_softmax/checkpoint/run5/decoder/encoder2000/decoder_model_e1999-i49999.ckpt")
    # K.set_session(sess)
    sess.run(init_op)

    dat = []

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    # ploti = []
    plote = []
    # plot_latent_var = []

    len1, len2 = datas.count_train_num_examples()
    x_test_len, y_test_len = datas.count_test_num_examples()

    global previous_latent
    global previous_latent_test
    global counter
    counter = 1
    latent_flag = True
    vae_obs = []
    vae_actions = []
    losses = []
    losses_test = []

    ploti = []
    for i in range(50016):
        ploti.append(i)

    try:
        for e in range(FLAGS.epochs):  # for e in range(601, FLAGS.epochs):
            print(range(FLAGS.epochs))
            counter = 1
            losses_per_epoch = []
            losses_per_epoch_test = []
            # ploti = []
            plot_latent_var = []
            plote.append(e)
            print(e, "/1999")
            for i in range(0, 50000, 32):  # 1563x32, 782x64 #50000 (0, 50000, 32)
                # print('i:', i)
                # Get Next Batch
                np_x, np_y = datas.train_next_batch(i)
                np_x_test, np_y_test = datas.test_next_batch(i)

                np_x = np.asarray(np_x)
                # print('np_x:', np_x)
                # print(np_x.shape)

                np_y = np.asarray(np_y)
                # print('np_y:', np_y)
                # print(np_y.shape)

                # Get Test Next Batch
                np_x_test = np.asarray(np_x_test)
                np_y_test = np.asarray(np_y_test)

                # if (counter % 1000 == 0 or counter == 1):
                if (counter == 1):
                    latent_flag = True
                else:
                    latent_flag = False

                # WITH_KL
                # _, np_loss = sess.run([train_op, loss], {enc_inputs: input_enc_batch2, y_holder: y_batch, dec_inputs: input_dec_batch, learning_rate: FLAGS.learning_rate, temperature: FLAGS.init_temp})
                # NO_KL
                if (latent_flag == True):
                    # print('hey')
                    _, np_loss, decoder_sess, logits_y_sess, dec_inp, enc_inpp_sess, latent_sess, encc_shape_sess = sess.run(
                        [train_op, loss, decoder_last_layer, logits_y, new_inp, enc_inpp, latents, encc_shape],
                        {inputs: np_x, latent: initial_latent,
                         y_holder: np_y,
                         learning_rate: FLAGS.learning_rate,
                         temperature: FLAGS.init_temp})

                    # np_loss2, decoder_sess2, logits_y_sess2, dec_inp2, enc_inpp_sess2, latent_sess2, encc_shape_sess2 = sess.run([loss, decoder_last_layer, logits_y, new_inp, enc_inpp, latents, encc_shape],
                    #                                                    {inputs: np_x_test, latent: initial_latent,
                    #                                                     y_holder: np_y_test,
                    #                                                     learning_rate: FLAGS.learning_rate,
                    #                                                     temperature: FLAGS.init_temp})
                    encoder_sess = np.asarray(latent_sess)
                    # encoder_sess2 = np.asarray(latent_sess2)

                else:
                    # print('hey2')
                    _, np_loss, decoder_sess, logits_y_sess, dec_inp, enc_inpp_sess, latent_sess, encc_shape_sess = sess.run(
                        [train_op, loss, decoder_last_layer, logits_y, new_inp, enc_inpp, latents, encc_shape],
                        {inputs: np_x, latent: encoder_sess,
                         y_holder: np_y,
                         learning_rate: FLAGS.learning_rate,
                         temperature: FLAGS.init_temp})

                    # np_loss2, decoder_sess2, logits_y_sess2, dec_inp2, enc_inpp_sess2, latent_sess2, encc_shape_sess2 = sess.run([loss, decoder_last_layer, logits_y, new_inp, enc_inpp, latents, encc_shape],
                    #                                                  {inputs: np_x_test, latent: encoder_sess2,
                    #                                                   y_holder: np_y_test,
                    #                                                   learning_rate: FLAGS.learning_rate,
                    #                                                   temperature: FLAGS.init_temp})
                    encoder_sess = np.asarray(latent_sess)
                    # encoder_sess2 = np.asarray(latent_sess2)

                # print('train_loss:', np_loss)
                losses_per_epoch.append(np_loss)
                # losses_per_epoch_test.append(np_loss_test)

                list = []
                for j in range(32):
                    # print('latent:', encoder_sess[j])
                    encoder_sess_plot = np.argmax(encoder_sess[j])
                    # list.append(encoder_sess_plot)
                    # print('list:', list)
                    # print('plot:', encoder_sess_plot)
                    plot_latent_var.append(encoder_sess_plot)

                encoder_sess = encoder_sess[31]
                encoder_sess = np.expand_dims(encoder_sess, 0)
                # print('E:', encoder_sess)
                # encoder_sess2 = encoder_sess2[31]
                # encoder_sess2 = np.expand_dims(encoder_sess2, 0)

                counter += 1

                if (i >= 49984):
                    losses_per_epoch_np = np.asarray(losses_per_epoch)
                    mean_losses_per_epoch = np.mean(losses_per_epoch_np)
                    print('train_loss:', mean_losses_per_epoch)
                    losses.append(mean_losses_per_epoch.tolist())

                    # losses_per_epoch_test_np = np.asarray(losses_per_epoch_test)
                    # mean_losses_per_epoch_test = np.mean(losses_per_epoch_test_np)
                    # print('val_loss:', mean_losses_per_epoch_test)
                    # losses_test.append(mean_losses_per_epoch_test.tolist())

                    ###########################################################################

                ## if i % 10000 == 1:
                if e % 100 == 0 and (i >= 49984):
                    encoder_path = saver.save(sess,
                                              FLAGS.checkpoint_dir + '\\encoder\\encoder_test\\encoder_model_e{}-i{}.ckpt'.format(
                                                  e, i))
                    decoder_path = dec_saver.save(sess,
                                                  FLAGS.checkpoint_dir + '\\decoder\\decoder_test\\decoder_model_e{}-i{}.ckpt'.format(
                                                      e, i))
                if e == 999 and (i >= 49984):  # (i % 1000 == 0 or i == 49999):
                    encoder_path = saver.save(sess,
                                              FLAGS.checkpoint_dir + '\\encoder\\encoder1000\\encoder_model_e{}-i{}.ckpt'.format(
                                                  e, i))
                    decoder_path = dec_saver.save(sess,
                                                  FLAGS.checkpoint_dir + '\\decoder\\decoder1000\\decoder_model_e{}-i{}.ckpt'.format(
                                                      e, i))
                elif e == 1999 and (i >= 49984):
                    encoder_path = saver.save(sess,
                                              FLAGS.checkpoint_dir + '\\encoder\\encoder2000\\encoder_model_e{}-i{}.ckpt'.format(
                                                  e, i))
                    decoder_path = dec_saver.save(sess,
                                                  FLAGS.checkpoint_dir + '\\decoder\\decoder2000\\decoder_model_e{}-i{}.ckpt'.format(
                                                      e, i))

                # ploti.append(i)

                print("e is:",e)
                print("i is:",i)
                print(ploti)
                print(plot_latent_var)
                if (e % 100 == 0 and i >= 49984) or (e == 999 and i >= 49984) or (e == 1999 and i >= 49984):
                    print("We are heeeeeere!!!!!")
                    plot_vae(ploti, plot_latent_var, e)
                    #plot_both_loss(plote, losses,losses_test, e)
                    plot_loss(plote, losses, e)

            FLAGS.init_temp = np.maximum(FLAGS.init_temp * np.exp(-FLAGS.anneal_rate * e), FLAGS.min_temp)
            # FLAGS.learning_rate *= 0.9

            print('Temperature updated to {}\n'.format(FLAGS.init_temp))

    except KeyboardInterrupt:
        print()

    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()

        print('Mean of Losses:', np.mean(losses))
        print('x_train_num:', len1, 'y_train:', len2, 'x_test_num:', x_test_len, 'y_test_num', y_test_len)
        # plot_vae(ploti, plot_latent_var)


def plot_vae(ploti, latent_var, e):
    ploti = np.asarray(ploti)
    latent_var = np.asarray(latent_var)
    e = np.asarray(e)
    for i in range(0, 50000, 1000):
        # print(i)
        if (i == 0):
            ploti2 = ploti[:i + 1000].copy()
            save_path = os.getcwd() + '\\results_gumbel_softmax\\plot\\hopper\\epoch{}-i{}_hopper_latents.png'.format(e, i)
            pred_context = latent_var[:i + 1000].copy()
        else:
            ploti2 = ploti[j:i + 1000].copy()
            save_path = os.getcwd() + '\\results_gumbel_softmax\\plot\\hopper\\epoch{}-i{}_hopper_latents.png'.format(e, i)
            pred_context = latent_var[j:i + 1000].copy()

        j = i + 1000
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(ploti2, pred_context, c='r', marker='X')
        ax.plot(ploti2, pred_context)
        plt.xlabel("TimeSteps")
        # plt.xticks([0, 1000 ,3000, 5000, 7000, 9000, 9500, 10000])
        # naming the y axis
        plt.ylabel("Latent Vars")
        # giving a title to my graph
        plt.title("Hopper-V2 (5 Modes)")
        plt.savefig(save_path, bbox_inches='tight')
        # plt.show()
        plt.clf()
        plt.close(fig)


def plot_vae_test(ploti, latent_var, e):
    ploti = np.asarray(ploti)
    latent_var = np.asarray(latent_var)
    ploti_size = ploti.size
    latent_var_size = latent_var.size
    e = np.asarray(e)

    ploti = ploti[:ploti_size].copy()
    save_path = os.getcwd() + '\\results_gumbel_softmax\\plot\\hopper\\epoch{}_hopper_latents.png'.format(e)
    pred_context = latent_var[:ploti_size].copy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(ploti, pred_context, c='r', marker='X')
    ax.plot(ploti, pred_context)
    plt.xlabel("TimeSteps")
    # plt.xticks([0, 1000 ,3000, 5000, 7000, 9000, 9500, 10000])
    # naming the y axis
    plt.ylabel("Latent Vars")
    # giving a title to my graph
    plt.title("Hopper-V2 (5 Modes)")
    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()


def plot_loss(epochs, loss, e):
    epochs = np.asarray(epochs)
    loss = np.asarray(loss)
    e = np.asarray(e)

    epochs = epochs.copy()
    save_path = os.getcwd() + '\\results_gumbel_softmax\\plot\\hopper\\epoch{}_hopper_loss.png'.format(e)
    pred_context = loss.copy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(epochs, pred_context, c='r', marker='X')
    ax.plot(epochs, pred_context)
    plt.xlabel("Epochs")
    # plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # naming the y axis
    plt.ylabel("Train Loss")
    # giving a title to my graph
    plt.title("Hopper-V2 (5 Modes)")
    plt.savefig(save_path, bbox_inches='tight')
   #plt.show()


def plot_both_loss(epochs, loss, loss_test, e):
    epochs = np.asarray(epochs)
    loss = np.asarray(loss)
    loss_test = np.asarray(loss_test)
    e = np.asarray(e)

    epochs = epochs.copy()
    save_path = os.getcwd() + '\\results_gumbel_softmax\\plot\\hopper\\epoch{}_hopper_loss.png'.format(e)
    pred_context = loss.copy()
    pred_context2 = loss_test.copy()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(epochs, pred_context, c='r', marker='X')
    ax.plot(epochs, pred_context, label="line 1", color="cornflowerblue")
    ax2.plot(epochs, pred_context2, label="line 2", color="Orange")
    ax.set_xlabel('Epochs')
    # plt.xlabel("Epochs")
    # plt.xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    # naming the y axis
    # plt.ylabel("Loss")
    ax.set_ylabel('Loss', color="cornflowerblue")
    ax2.set_ylabel('Test Loss', color="Orange")
    # giving a title to my graph
    plt.title("Hopper-V2")
    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()


def renameVAE():
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

    # RENAME MAP LAYERS-BIASES-GRAPHS
    OLD_CHECKPOINT_FILE = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\encoder\\encoder2000\\encoder_model_e1999-i49984.ckpt'  # '\\encoder\\encoder_model49599.ckpt'
    NEW_CHECKPOINT_FILE = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\test\\encoder\\encoder_model_e2000.ckpt'  # '\\encoder_new(2)\\encoder_model_new49600.ckpt'
    # OLD_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model9999.ckpt'
    # NEW_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model_new/model_new9999.ckpt'

    reader = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE)
    shape_from_key = reader.get_variable_to_shape_map()
    dtype_from_key = reader.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')
    # print(dtype_from_key)

    vars_to_rename = {
        "encoder_h1/kernel": "Encoder/dense/kernel",
        "encoder_h1/bias": "Encoder/dense/bias",
        "encoder_h2/kernel": "Encoder/dense_1/kernel",
        "encoder_h2/bias": "Encoder/dense_1/bias",
        "encoder_out/bias": "Encoder/dense_2/bias",
        "encoder_out/kernel": "Encoder/dense_2/kernel",

        # "_CHECKPOINTABLE_OBJECT_GRAPH": "lstm/basic_lstm_cell/bias",
    }
    new_checkpoint_vars = {}
    reader1 = tf.compat.v1.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
    for old_name in reader1.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader1.get_tensor(old_name))

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(new_checkpoint_vars)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver.save(sess, NEW_CHECKPOINT_FILE)

    # Read New_Checkpoint
    reader2 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE)
    shape_from_key2 = reader2.get_variable_to_shape_map()
    dtype_from_key2 = reader2.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)

    OLD_CHECKPOINT_FILE2 = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\decoder\\decoder2000\\decoder_model_e1999-i49984.ckpt'  # 1999-49999.ckpt'
    NEW_CHECKPOINT_FILE2 = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\test\\decoder\\decoder_model_e2000.ckpt'  # 2000_new50000.ckpt'

    reader2 = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE2)
    shape_from_key = reader2.get_variable_to_shape_map()
    dtype_from_key = reader2.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')

    vars_to_rename = {
        "decoder_h1/kernel": "Policy/dense_3/kernel",
        "decoder_h1/bias": "Policy/dense_3/bias",
        "decoder_h2/kernel": "Policy/dense_4/kernel",
        "decoder_h2/bias": "Policy/dense_4/bias",
        "decoder_out/bias": "Policy/dense_5/bias",
        "decoder_out/kernel": "Policy/dense_5/kernel",

    }
    new_checkpoint_vars = {}
    reader3 = tf.compat.v1.train.NewCheckpointReader(OLD_CHECKPOINT_FILE2)
    for old_name in reader3.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader3.get_tensor(old_name))

    init = tf.compat.v1.global_variables_initializer()
    saver2 = tf.compat.v1.train.Saver(new_checkpoint_vars)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver2.save(sess, NEW_CHECKPOINT_FILE2)

    # Read New_Checkpoint
    reader4 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE2)
    shape_from_key2 = reader4.get_variable_to_shape_map()
    dtype_from_key2 = reader4.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)


def rename():
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

    # RENAME MAP LAYERS-BIASES-GRAPHS
    OLD_CHECKPOINT_FILE = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\encoder\\encoder2000\\encoder_model_e1999-i49984.ckpt'  # '\\encoder\\encoder_model49599.ckpt'
    NEW_CHECKPOINT_FILE = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\trpo_plugins\\encoder\\encoder_model_e2000.ckpt'  # '\\encoder_new(2)\\encoder_model_new49600.ckpt'
    # OLD_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model9999.ckpt'
    # NEW_CHECKPOINT_FILE = FLAGS.checkpoint_dir + '/model_new/model_new9999.ckpt'

    reader = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE)
    shape_from_key = reader.get_variable_to_shape_map()
    dtype_from_key = reader.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')
    # print(dtype_from_key)

    vars_to_rename = {
        "encoder_h1/kernel": "Encoder/dense_6/kernel",
        "encoder_h1/bias": "Encoder/dense_6/bias",
        "encoder_h2/kernel": "Encoder/dense_7/kernel",
        "encoder_h2/bias": "Encoder/dense_7/bias",
        "encoder_out/bias": "Encoder/dense_8/bias",
        "encoder_out/kernel": "Encoder/dense_8/kernel",

        # "_CHECKPOINTABLE_OBJECT_GRAPH": "lstm/basic_lstm_cell/bias",
    }
    new_checkpoint_vars = {}
    reader1 = tf.compat.v1.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
    for old_name in reader1.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader1.get_tensor(old_name))

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(new_checkpoint_vars)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver.save(sess, NEW_CHECKPOINT_FILE)

    # Read New_Checkpoint
    reader2 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE)
    shape_from_key2 = reader2.get_variable_to_shape_map()
    dtype_from_key2 = reader2.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)

    OLD_CHECKPOINT_FILE2 = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\decoder\\decoder2000\\decoder_model_e1999-i49984.ckpt'  # 1999-49999.ckpt'
    NEW_CHECKPOINT_FILE2 = '.\\results_gumbel_softmax\\checkpoint\\run9(5modes)' + '\\trpo_plugins\\decoder\\decoder_model_e2000.ckpt'  # 2000_new50000.ckpt'

    reader2 = tf.train.load_checkpoint(OLD_CHECKPOINT_FILE2)
    shape_from_key = reader2.get_variable_to_shape_map()
    dtype_from_key = reader2.get_variable_to_dtype_map()
    print(shape_from_key)
    print('')
    print(sorted(shape_from_key.keys()))
    print('')

    vars_to_rename = {
        "decoder_h1/kernel": "Policy/dense_9/kernel",
        "decoder_h1/bias": "Policy/dense_9/bias",
        "decoder_h2/kernel": "Policy/dense_10/kernel",
        "decoder_h2/bias": "Policy/dense_10/bias",
        "decoder_out/bias": "Policy/dense_11/bias",
        "decoder_out/kernel": "Policy/dense_11/kernel",
    }

    new_checkpoint_vars = {}
    reader3 = tf.compat.v1.train.NewCheckpointReader(OLD_CHECKPOINT_FILE2)
    for old_name in reader3.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader3.get_tensor(old_name))

    init = tf.compat.v1.global_variables_initializer()
    saver2 = tf.compat.v1.train.Saver(new_checkpoint_vars)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver2.save(sess, NEW_CHECKPOINT_FILE2)

    # Read New_Checkpoint
    reader4 = tf.train.load_checkpoint(NEW_CHECKPOINT_FILE2)
    shape_from_key2 = reader4.get_variable_to_shape_map()
    dtype_from_key2 = reader4.get_variable_to_dtype_map()
    print(shape_from_key2)
    print('')
    print(sorted(shape_from_key2.keys()))
    # print(dtype_from_key2)


# TESTING

class ClonosEncoder(object):
    @staticmethod
    def create_encoder(observation_dimensions, latent_dimensions, action_dimensions):
        with tf.compat.v1.name_scope('Encoder'):
            encoder_x = Input(shape=(observation_dimensions + latent_dimensions,), dtype=tf.float32)

            l1 = Dense(100, activation='tanh')(
                encoder_x)  # name='dense_6'), activation=tf.keras.layers.LeakyReLU(alpha=0.01)
            l2 = Dense(100, activation='tanh')(l1)  # name='dense_7')

            out = Dense(action_dimensions, activation='softmax')(l2)  # name='dense_8')

        model = Model(inputs=encoder_x, outputs=out)

        return model, encoder_x, l1, l2


class Policy(object):
    @staticmethod
    def create_policy(observation_dimensions, latent_dimensions, action_dimensions):
        with tf.compat.v1.name_scope('Policy'):
            x = Input(shape=(observation_dimensions + latent_dimensions,), dtype=tf.float32)

            h = Dense(100, activation='tanh')(x)  # , name='Policy/dense_9')(x)
            h1 = Dense(100, activation='tanh')(h)  # , name='Policy/dense_10')(h)

            out = Dense(action_dimensions)(h1)  # , name='Policy/dense_11')(h1)

        model = Model(inputs=x, outputs=out)

        return model, x, h, h1


class Agent(object):
    def __init__(self, observation_dimensions=11, latent_dimensions=5, action_dimensions=3):
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.latent_dimensions = latent_dimensions
        self.sess = tf.compat.v1.Session(config=config)

        self.model2, self.encoder_x, self.l1, self.l2 = ClonosEncoder.create_encoder(self.observation_dimensions,
                                                                                     self.latent_dimensions,
                                                                                     self.action_dimensions)
        self.model, self.x, self.h, self.h1 = Policy.create_policy(self.observation_dimensions, self.latent_dimensions,
                                                                   self.action_dimensions)

        self.encoder_logits = self.model2.outputs[0]
        self.logits = self.model.outputs[0]
        self.env = gym.make("Hopper-v2")

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def one_hot_encoding(self, x):
        # print('OneHotEncoding...')
        argmax = np.argmax(x)
        self.plot_test_latent_var.append(argmax)
        encoded = to_categorical(argmax, num_classes=5)
        # print('enc:', encoded)
        return encoded.tolist()

    def init_encoder(self, observation):
        print ("init encoder..........")
        # init_latent = [1, 0, 0]
        init_latent = [1, 0, 0, 0, 0]
        init_latent = np.asarray(init_latent)
        enc_input = np.concatenate((np.asarray(observation), init_latent))
        print("enc_input is --> ", enc_input)
        # print('obs_l:', enc_input)
        mu2 = self.sess.run(self.encoder_logits, feed_dict={self.encoder_x: [enc_input]})[0]
        latent_prob = np.asarray(mu2)
        #print("latent_prob is--> ", latent_prob)
        latent = self.one_hot_encoding(latent_prob)
        #print ("latent now is-->", latent)

        decoder_input = np.concatenate((np.asarray(observation), latent))
        return decoder_input, latent

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
        latent_new = self.one_hot_encoding(latent_prob)

        # array, shape = self.latent_sequence(mu2)

        # one_hot_enc = self.keras_oneHotEncoder(array, shape)

        # exit(0)

        global_concat_test = latent_new
        decoder_input = np.concatenate((np.asarray(observation), latent_new))
        return decoder_input, latent_new

    def act_test(self, decoder_input):  # , latent):
        mu = self.sess.run(self.logits, feed_dict={self.x: [decoder_input]})[0]
        act = mu
        return act

    def test(self):
        enc_saver = tf.compat.v1.train.Saver(self.model2.weights)
        enc_saver.restore(self.sess,
                          "results_gumbel_softmax\\checkpoint\\run9(5modes)\\test\\encoder\\encoder_model_e2000.ckpt")
        saver = tf.compat.v1.train.Saver(self.model.weights)
        saver.restore(self.sess, "results_gumbel_softmax\\checkpoint\\run9(5modes)\\test\\decoder\\decoder_model_e2000.ckpt")

        actions = []
        obs = []
        discounted_rewards = []
        total_rewards = []
        print('Episodes,Reward')
        self.init = True

        for i_episode in range(50):#50
            self.counter2 = 0
            self.plot_test_latent_var = []
            self.plot_i_test = []
            steps = 0
            r_i = []
            observation = self.env.reset()
            #print('obs:',observation)
            dec_input1, latent_init = self.init_encoder(observation)

            total_reward = 0

            for t in range(1000):#1000
                if self.init:
                    print('latent:', latent_init)
                    print('dec_input:', dec_input1)
                    obs.append(dec_input1)
                    #print ('observations now.....', obs)
                    action = self.act_test(dec_input1)
                    #print ('new action ...', action)
                    action = action.tolist()
                    #print ('new action TOLIST...', action)
                else:
                    obs.append(dec_input2)
                    #print ('observations now.....', obs)
                    action = self.act_test(dec_input2)
                    action = action.tolist()

                self.counter2 += 1
                print("counter2 is...", self.counter2)
                self.env.render()
                observation2, reward, done, _ = self.env.step(action)
                #print('obs2:',observation2)

                if self.init:
                    dec_input2, latent = self.new_encoder(observation2, latent_init)
                    #print('latent:',latent)
                   #print('dec_in2:',dec_input2)
                else:
                    dec_input2, latent = self.new_encoder(observation2, latent)
                    # print('latent_2:',latent)
                    # print('dec_in2_2:',dec_input2)

                steps += 1
                r_i.append(reward)
                actions.append(action)
                total_reward += reward
                # action = unnormalize_action(action)

                self.plot_i_test.append(t)
                self.init = False
                if t % 100 == 0:
                    print("%i/%i" % (t + 100, 1000))
                if t >= 1000 or done:
                    # if done:
                    #    print('')
                    # exit(0)
                    # continue

                    np1 = np.asarray(self.plot_i_test)
                    np2 = np.asarray(self.plot_test_latent_var)
                    print('np1_s:', np1.size)
                    print('np2_s:', np2.size)
                    plot_vae_test(self.plot_i_test, self.plot_test_latent_var, i_episode)
                    break

            print('{0},{1}'.format(i_episode, total_reward))
            # exit(0)
            discounted_rewards.extend(discount(r_i, 0.995))
            total_rewards.append(total_reward)

            # self.env.close()
            # self.sess.close()
        return actions, obs, discounted_rewards, total_rewards


def main():
    if tf.io.gfile.exists(FLAGS.log_dir):
        tf.io.gfile.rmtree(FLAGS.log_dir)
    tf.io.gfile.makedirs(FLAGS.log_dir)
    tf.io.gfile.makedirs(FLAGS.data_dir)
    tf.io.gfile.makedirs(FLAGS.checkpoint_dir)
    tf.io.gfile.makedirs(FLAGS.results_dir)

    train()
    print("Vae model trained")

    # RENAME LAYERS
    renameVAE()
    rename()

    # Testing
    agent = Agent()
    actions_vae, obs_vae, discounted_rewards_vae, total_rewards_vae = agent.test()
    print('Sum of Rewards VAE:', sum(total_rewards_vae))  # np.mean(total_rewards_vae)
    print('Mean Reward VAE:', np.mean(total_rewards_vae))


if __name__ == "__main__":
    main()