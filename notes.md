## Tensorflow logging:

```PowerShell
$env:TF_CPP_MIN_LOG_LEVEL=1
```



```sh
set -gx TF_CPP_MIN_LOG_LEVEL 1
```

 Level | Level for Humans | Level Description
-------|------------------|------------------------------------
 0     | DEBUG            | [Default] Print all messages
 1     | INFO             | Filter out INFO messages
 2     | WARNING          | Filter out INFO & WARNING messages
 3     | ERROR            | Filter out all messages


$env:TF_GPU_ALLOCATOR=cuda_malloc_async


## CUPTI

Install CUDA (same version as conda installed), find `cupti64_*.dll` in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64`, copy to `<conda env base dir>\Library\bin`

From stackoverflow:
> On Nvidia Control Panel, there is a Developer / Manage GPU Performance Counters section. Default toggle is to limit access to GPU preformance counters to admin users only. But you must select 'Allow acces to the GPU prformance counters to all users'. Once toggled, access permissions to the cupti dll are resolved. –

## Check if GPU available to Tensorflow:

Activate TF conda env first

```PowerShell
conda activate audio
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```

```Python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```


## LD PRELOAD

```sh
set -gx LD_LIBRARY_PATH "$LD_LIBRARY_PATH:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt"
set -gx XLA_FLAGS "--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib"
set -gx TF_GPU_ALLOCATOR cuda_malloc_async
```

Made symlinks in `/home/emredjan/conda/envs/tf/lib/python3.10/site-packages/tensorrt`

- `libnvinfer.so.7 -> libnvinfer.so.8`
- `libnvinfer_plugin.so.7 -> libnvinfer_plugin.so.8`

Install nvcc

- `conda install -c nvidia cuda-nvcc`


conda install -c conda-forge ncurses #(may need specific build)

Copy lib

```sh
cd /home/emredjan/conda/envs/tf/lib
mkdir nvvm
mkdir nvvm/libdevice
cp libdevice.10.bc nvvm/libdevice/
```
