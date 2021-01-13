import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)




##from keras import backend
##backend.tensorflow_backend._get_available_gpus() FAIL
##
##from tensorflow.python.client import device_lib
##print(device_lib.list_local_devices())


##from tensorflow.python.client import device_lib
##def get_available_devices():
##    local_device_protos = device_lib.list_local_devices()
##    return [x.name for x in local_device_protos]
##print(get_available_devices()) 

##import tensorflow as tf    FAILS
##sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K
print(tf.test.gpu_device_name())
print(tf.config.list_physical_devices('GPU'))
print(device_lib.list_local_devices())
##print(K.tensorflow_backend._get_available_gpus())
