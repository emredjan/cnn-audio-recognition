import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.client import device_lib

devices = {dev.device_type: dev.physical_device_desc for dev in device_lib.list_local_devices()}  # type: ignore

if 'GPU' in devices.keys():
    print('Tensorflow: GPU available, will use the following device for calculation:')
    print(f"\t{devices.get('GPU')}")
elif 'CPU' in devices.keys():
    print('Tensorflow: Only CPU available, no GPU calculation possible.')
else:
    print('Tensorflow: No compatible device found.')
