import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.python.client import device_lib

devices = {dev.device_type: dev.physical_device_desc for dev in device_lib.list_local_devices()}  # type: ignore

if 'GPU' in devices.keys():
    print('Tensorflow: GPU available, will use the following device for calculation:')
    print(f"\t{devices.get('GPU')}")
elif 'CPU' in devices.keys():
    print('Tensorflow: Only CPU available, no GPU calculation possible.')
else:
    print('Tensorflow: No compatible device found.')


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5292)]
    )
