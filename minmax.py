import numpy as np
import pandas as pd

actions = pd.read_csv('./dataset/hopper-v2/hopper_v2_actions(50016).csv')
observ = pd.read_csv('./dataset/hopper-v2/hopper_v2_observations(50016).csv')

print(actions)
print(observ)
actions_np = np.asarray(actions)
observ_np = np.asarray(observ)

min_act = np.min(actions_np)
max_act = np.max(actions_np)

min_obs = np.min(observ_np)
max_obs = np.max(observ_np)

print('actions..min:',min_act,' max:',max_act)
print('obs..min:',min_obs,' max:',max_obs)