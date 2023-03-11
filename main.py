from myconfig import myconfig
#from environment import
#from environment import hopper
import gym
import os
os.add_dll_directory("C:\\Users\\ETAPTA\\PycharmProjects\\HopperProject\\venv\\Lib\\site-packages\\mujoco_py\\binaries\\windows\\mujoco210\\bin")
import mujoco_py
from statistics import mean
from sklearn.model_selection import KFold
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
#tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import VAE.vae_gumbel_softmax as VAE
from sklearn import preprocessing
import numpy as np
import trpo as trpo
#from tensorflow.python.keras.utils import plot_model
from keras.utils.vis_utils import plot_model

config = tf.ConfigProto()
#tf.compat.v1.enable_eager_execution(config)
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
not_trained = False

def bcloning(x_train, y_train, x_test, y_test):
    model = agent.model
    x = agent.x

    sess = tf.Session(config=config)
    checkpoint_path = "./checkpoint/bcloning.ckpt"
    # discriminator_path = myconfig['output_dir']+"output/bcloning_discriminator.ckpt"
    predictions = model.outputs[0]
    labels = tf.placeholder(tf.float32, shape=(None), name='y')
    loss = tf.reduce_mean(tf.square(predictions - labels))
    opt = tf.train.AdamOptimizer().minimize(loss)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(model.weights)
    # discriminator_saver = tf.train.Saver(agent.discriminator.discriminate.weights)
    # saver.restore(sess, checkpoint_path)

    epochs = myconfig['bcloning_epochs']
    batch_size=64
    kf = KFold(n_splits=myconfig['bcloning_folds'], shuffle=True)
    count = 0
    print(kf.get_n_splits(x_train))
    # step = 0
    for train_index, validation_index in kf.split(x_train):
        count += 1
        x_t = x_train[train_index]
        x_val = x_train[validation_index]
        y_t = y_train[train_index]
        y_val = y_train[validation_index]
        print('New Fold',count)
        for i in range(epochs):
            train_loss = []
            # preds = []
            for j in range(0, len(x_t), batch_size):
                if len(x_t[j:j+batch_size]) < 64:
                    continue
                _, loss_run, _ = sess.run([opt, loss, labels],
                                                           feed_dict={x: x_t[j:j + batch_size],
                                                                      labels: y_t[j:j + batch_size]})
                train_loss.append(loss_run)

            val_loss_run, _ = sess.run([loss, labels],
                                          feed_dict={x: x_val,
                                                     labels: y_val})
            print('epoch', i, 'train loss', mean(train_loss), 'validation_loss', val_loss_run)

    saver.save(sess, checkpoint_path)
    test_loss_run, _ = sess.run([loss, labels],
                               feed_dict={x: x_test,
                                          labels: y_test})

    print(test_loss_run)


#------------------------------------------------------------------------------------------------------#

def pandas1(obs):
    df_norm_obs = pd.DataFrame(obs, columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10','var11'])
    return df_norm_obs

def pandas2(actions):
    df_norm = pd.DataFrame(actions, columns=['action1', 'action2', 'action3'])
    return df_norm

def apply_normalization_observations(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm_obs = pd.DataFrame(np_scaled, columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var11'])
    return df_norm_obs

def apply_normalization_actions(df):
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(df)
    df_norm = pd.DataFrame(np_scaled, columns=['action1', 'action2', 'action3'])
    return df_norm
#obs = pandas1(obs)
#actions = pandas2(actions)

#Load the Dataset
print('Loading Dataset...')

#x_train = obs[:50000]
#x_test = obs[50000:]
#y_train = actions[:50000]
#y_test = actions[50000:]

#x_train = apply_normalization_observations(x_train)
#x_test = apply_normalization_observations(x_test)
#print('obs_normalized:')
#print(obs_normalized)
#y_train = apply_normalization_actions(y_train)
#y_test = apply_normalization_actions(y_test)
#print('actions_normalized:')
#print(actions_normalized)

obs = pd.read_csv('../dataset/hopper-v2/hopper_v2_observations(50016).csv')
obs_test = pd.read_csv('../dataset/hopper-v2/hopper_v2_observations(62000).csv')
obs_train = np.asarray(obs)
obs_test = np.asarray(obs_test)
print('OBS:', obs_train.shape)
obs_test = obs_test[:12000]
print('OBS_TEST:', obs_test.shape)
actions = pd.read_csv('../dataset/hopper-v2/hopper_v2_actions(50016).csv')
actions_test = pd.read_csv('../dataset/hopper-v2/hopper_v2_actions(62000).csv')
actions_train = np.asarray(actions)
actions_test = np.asarray(actions_test)
#actions_train = actions[:50000]
actions_test = actions_test[:12000]

#x_train, x_test, y_train, y_test = train_test_split(obs, actions, test_size=0.2, shuffle=True)
#obs_train, actions_train = shuffle1(obs_train, actions_train)
#obs_test, actions_test = shuffle1(obs_test, actions_test)

#EDW PANDAS
#x_train = pandas1(obs_train)
#x_test = pandas1(obs_test)
#y_train = pandas2(actions_train)
#y_test = pandas2(actions_test)
x_train = obs_train
x_test = obs_test
y_train = actions_train
y_test = actions_test

print(len(x_train),len(x_test), len(y_train), len(y_test))

def plot_discriminator():
    discriminator = trpo.Discriminator(action_dimensions=3, observation_dimensions=11)
    plot_model(discriminator.discriminator, show_shapes=True, to_file='discriminator_model.png')
    plot_model(discriminator.discriminate, show_shapes=True, to_file='discriminate_model.png')

def test_discriminator(observations_test_expert, actions_test_expert, observations_test_generator, actions_test_generator, gen_file, expert_file):
    saver = tf.train.Saver(agent.discriminator.discriminate.weights)
    saver.restore(agent.discriminator.sess, "./.output/exp0discriminator.ckpt")
    # expert_pred = np.exp(agent.discriminator.predict(observations_test_expert,actions_test_expert))
    expert_reward = -agent.discriminator.predict(observations_test_expert, actions_test_expert)
    # generator_pred = np.exp(agent.discriminator.predict(observations_test_generator,actions_test_generator))
    generator_reward = -agent.discriminator.predict(observations_test_generator, actions_test_generator)
    np.savetxt(expert_file, expert_reward, delimiter=",", header='expert_reward', comments='')
    np.savetxt(gen_file, generator_reward, delimiter=",", header='generator_reward', comments='')
    print("Expert avg:", np.mean(np.exp(-expert_reward)))
    print("Expert std:", np.std(np.exp(-expert_reward)))
    print("Expert min:", np.min(np.exp(-expert_reward)))
    print("Expert max:", np.max(np.exp(-expert_reward)))
    print("Generator avg:", np.mean(np.exp(-generator_reward)))
    print("Generator std:", np.std(np.exp(-generator_reward)))
    print("Generator min:", np.min(np.exp(-generator_reward)))
    print("Generator max:", np.max(np.exp(-generator_reward)))

def plot_discriminator_predict():
    sns.set(rc={'figure.figsize':(18,10)})
    expert = pd.read_csv("expert_train_reward_"+str(myconfig['exp'])+".csv")
    expert_test = pd.read_csv("expert_test_reward_"+str(myconfig['exp'])+".csv")
    generator = pd.read_csv("generator_train_reward_"+str(myconfig['exp'])+".csv")
    generator_test = pd.read_csv("generator_test_reward_"+str(myconfig['exp'])+".csv")

    df = pd.concat([expert, expert_test, generator, generator_test],axis=1)
    df.columns = ['expert_reward', 'expert_test_reward', 'generator_reward', 'generator_test_reward']
    df_pow = np.exp(-df)

    sns.distplot(df_pow['expert_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_train_set'})
    sns.distplot(df_pow['expert_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_test_set'})
    sns.distplot(df_pow['generator_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_train_set'})
    sns.distplot(df_pow['generator_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_test_set'})
    plt.savefig(myconfig['plot_dir']+'/discriminator_all.png')
    plt.clf()
    sns.distplot(df_pow['expert_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_train_set'},color='cyan')
    plt.savefig(myconfig['plot_dir']+'/discriminator_expert_trainset.png')
    plt.clf()
    sns.distplot(df_pow['expert_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'expert_test_set'},color='darkorange')
    plt.savefig(myconfig['plot_dir']+'/discriminator_expert_testset.png')
    plt.clf()
    sns.distplot(df_pow['generator_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_train_set'},color='green')
    plt.savefig(myconfig['plot_dir']+'/discriminator_generator_traintset.png')
    plt.clf()
    sns.distplot(df_pow['generator_test_reward'].dropna(), axlabel='discriminator prediction', kde_kws={'label':'generator_test_set'},color='red')
    plt.savefig(myconfig['plot_dir']+'/discriminator_generator_testset.png')


def discriminator_load_test():
    for files in [['./exp0_0%_test_GAIL_results.csv',#obs,
                   './exp0_0%_test_GAIL_actions_results.csv',#actions,
                   'generator_test_reward_' + str(myconfig['exp']) + '.csv',
                   'expert_test_reward_' + str(myconfig['exp']) + '.csv', x_test, y_test],
                  [str(myconfig['exp']) + '_0_observation_log.csv',
                   str(myconfig['exp']) + '_0_action_log.csv',
                   'generator_train_reward_' + str(myconfig['exp']) + '.csv',
                   'expert_train_reward_' + str(myconfig['exp']) + '.csv', x_train, y_train]]:

        generator_observations = pd.read_csv(files[0]).drop(['episode', 'latent1', 'latent2', 'latent3'], axis=1)  # files[0]
        #generator_observations = pd.read_csv(files[0]).drop(['episode'], axis=1)#files[0]
        generator_actions = pd.read_csv(files[1]).drop(['episode'], axis=1)#files[1]

        generator_actions = generator_actions.values
        generator_observations = generator_observations.values

        log_obs = pd.read_csv('./runs/Directed-INFO GAIL/run3/0_0_observation_log.csv').drop(['episode', 'latent1', 'latent2', 'latent3'], axis=1)
        log_act = pd.read_csv('./runs/Directed-INFO GAIL/run3/0_0_action_log.csv').drop(['episode'], axis=1)

        #print('gen_actions:', len(generator_actions))
        #print('gen_obs:', len(generator_observations))
        print('########################################################')
        test_discriminator(log_obs, log_act, generator_observations, generator_actions, files[2], files[3])
    print('########################################################')
    plot_discriminator_predict()

#x_train = pandas1(x_train)
#x_test = pandas1(x_test)
#y_train = pandas2(y_train)
#y_test = pandas2(y_test)

#actions_train = y_train.values
#obs_train = x_train.values
#actions_test = y_test.values
#obs_test = x_test.values
# obs_validate = obs_validate.values
# actions_validate = actions_validate.values

env = gym.make("Hopper-v2")
#env.reset()
observation_space = env.observation_space
action_space = env.action_space
observation_space_Dim = observation_space.shape[0]
action_space_Dim = action_space.shape[0]

print(observation_space, observation_space_Dim)
print(action_space, action_space_Dim)


#agent = trpo_plugins.TRPOAgent(env, action_dimensions=action_space_Dim, observation_dimensions=observation_space_Dim)
agent = trpo.TRPOAgent(env, action_dimensions=action_space_Dim, latent_dimensions=5, observation_dimensions=observation_space_Dim)

print("Starting Training")

#IF BEHAVIORAL CLONING
#bcloning(obs_train, actions_train, obs_test, actions_test)

#Train Agent

agent.train(obs_train, actions_train)
fname = '0%_test'
#Test Agent
#actions_bc, obs_bc, discounted_rewards_bc, total_rewards_bc = agent.run(50, vae=False, bcloning=True, fname=fname)
actions_vae, obs_vae, discounted_rewards_vae, total_rewards_vae = agent.run(50, vae=True, bcloning=False, fname=fname)#50
actions_gail, obs_gail, discounted_rewards_gail, total_rewards_gail = agent.run(50, vae=False, bcloning=False, fname=fname)#50

print('obs_train:', len(obs_train), 'actions_train:', len(actions_train), 'obs_test:', len(obs_test), 'actions_test:', len(actions_test))
#print('actions_BC:', len(actions_bc), 'obs_BC:', len(obs_bc))
print('actions_VAE:', len(actions_vae), 'obs_VAE:', len(obs_vae))
print('actions_gail:', len(actions_gail), 'obs_gail:', len(obs_gail), '\n')

#print('Sum of Rewards BC:', sum(total_rewards_bc))#np.mean(total_rewards_vae)
print('Sum of Rewards VAE:', sum(total_rewards_vae))#np.mean(total_rewards_vae)
print('Sum of Rewards Directed-Info Gail:', sum(total_rewards_gail))#np.mean(total_rewards_gail)

#print('Max Reward BC:', np.mean(total_rewards_bc))
print('Mean Reward VAE:', np.mean(total_rewards_vae))
print('Mean Reward Directed-Info Gail:', np.mean(total_rewards_gail))

#print(actions_gail)
#print(obs_gail)

#Load Discriminator
#discriminator_load_test()

env.close()