import os
os.add_dll_directory("C:\\Users\\ETAPTA\\PycharmProjects\\HopperProject\\venv\\Lib\\site-packages\\mujoco_py\\binaries\\windows\\mujoco210\\bin")
#os.add_dll_directory("C:\\Users\\ETAPTA\\mjpro131\\bin")
import gym
from gym import wrappers
from expert_data.load_policy import load_policy
import pickle
#import tensorflow as
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import expert_data.tf_util as tf_util
import numpy as np
import pandas as pd
import mujoco_py

print("gym version")
print(gym.__version__)
print("mujoco_py version")
#print(mujoco_py.__version__)

print("tf version")
print(tf.__version__)


# Open Expert Hopper Policy File
def open_policy(filename):
    with open(filename, 'rb') as f:
        print('---[Data] :')
        data = pickle.loads(f.read())
open_policy('Hopper-v2.pkl')

#RUN THE EXPERT TO CREATE DATASET
def generate_data(expert_policy_file, env_name, output_dir=None, save=False, max_timesteps=1000):
    num_rollouts = 50
    print('loading and building expert policy')
    policy_fn = load_policy(expert_policy_file)
    #policy_fn = load_policy2(expert_policy_file)

    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(env_name)
        max_steps = max_timesteps#env.spec.timestep_limit

        if save:
            expert_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'expert')
            env = wrappers.Monitor(env, expert_results_dir, force=True)

        returns = []
        observations = []
        actions = []
        render=True
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            steps = 0
            total_rewards = 0.

            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                env.render()
                obs, reward, done, _ = env.step(action)
                total_rewards += reward
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(total_rewards)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'mean_return': np.mean(returns),
                       'std_return': np.std(returns)}

        if output_dir != 'None':
            #output_dir = os.path.join(os.getcwd(), output_dir)
            output_dir = 'dataset/'
            filename = '{}_data_{}_rollouts.pkl'.format(env_name, num_rollouts)
            with open(output_dir + '/' + filename, 'wb') as f:
                pickle.dump(expert_data, f)

generate_data('./Hopper-v2.pkl', 'Hopper-v2', './dataset')#hw1/experts/Hopper-v1.pkl

expert_data = './dataset/Hopper-v2_data_50_rollouts.pkl'
with open(expert_data, 'rb') as f:
    data = pickle.loads(f.read())

obs = np.stack(data['observations'], axis=0)
actions = np.squeeze(np.stack(data['actions'], axis=0))

def pandas1(obs):
    df_norm_obs = pd.DataFrame(obs, columns=['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10','var11'])
    return df_norm_obs

def pandas2(actions):
    df_norm = pd.DataFrame(actions, columns=['action1', 'action2', 'action3'])
    return df_norm

obs.tofile("./dataset/hopper-v2/obs_v2(50000).txt", sep=',', format=("%f"))
actions.tofile("./dataset/hopper-v2/actions_v2(50000).txt", sep=',', format=("%f"))

obs = pandas1(obs)
actions = pandas2(actions)

obs.to_csv('./dataset/hopper-v2/hopper_v2_observations(50000).csv', index=False)
actions.to_csv('./dataset/hopper-v2/hopper_v2_actions(50000).csv', index=False)
print("Observations: \n", obs)
print("Actions: \n", actions)