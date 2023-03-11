import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp

print(cp.list_variables('.\\VAE\\results_gumbel_softmax\\checkpoint\\run9(5modes)\\trpo_plugins\\decoder\\decoder_model_e2000.ckpt'))
print(cp.list_variables('.\\VAE\\results_gumbel_softmax\\checkpoint\\run9(5modes)\\trpo_plugins\\encoder\\encoder_model_e2000.ckpt'))

