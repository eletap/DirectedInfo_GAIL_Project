import gym
from gym import wrappers
from expert_data.load_policy import load_policy
import pickle
import tensorflow as tf
import expert_data.tf_util as tf_util
import os
import numpy as np
import pandas as pd



with open('Hopper-v2.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.get("GaussianPolicy").get("obsnorm"))#nonlin_type