from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

checkpoint_path = tf.train.latest_checkpoint('/home/iouiwc/github/ICSTN/MNIST-tensorflow/models_0')
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name", key)
    print("value", reader.get_tensor(key))