"""
import tensorflow as tf
import gym
import numpy as np
print("tf version")
print(tf.__version__)
print(gym.__version__)
print(np.__version__)
import os
os.add_dll_directory("C:\\Users\\ETAPTA\\PycharmProjects\\HopperProject\\venv\\Lib\\site-packages\\mujoco_py\\binaries\\windows\\mujoco210\\bin")

import mujoco_py

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
rank_1_tensor_lst = rank_1_tensor.tolist()
print(rank_1_tensor_lst)
print(type(rank_1_tensor_lst))



a = tf.Variable(5, name="a")
b = tf.Variable(7, name="b")
c = (b**2 - a**3)**5
print(c)
print(type(c))
"""
import matplotlib.pyplot  as plt
import numpy as np
xpoints=[]
xpoints.append(1)
xpoints.append(2)
xpoints.append(4)
xpoints.append(8)
xpoints.append(16)
ypoints=range(0,len(xpoints))

plt.plot(xpoints)
#plt.show()

plt.savefig('foo.png')