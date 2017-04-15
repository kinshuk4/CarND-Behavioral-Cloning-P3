# Simple Sequential Model

6428/6428 [==============================] - 21s - loss: 8775679.6030 - val_loss: 5997.5624
Epoch 2/7
6428/6428 [==============================] - 20s - loss: 4140.5004 - val_loss: 3072.8848
Epoch 3/7
6428/6428 [==============================] - 25s - loss: 2750.3284 - val_loss: 2642.8367
Epoch 4/7
6428/6428 [==============================] - 37s - loss: 1843.7224 - val_loss: 1684.3496
Epoch 5/7
6428/6428 [==============================] - 39s - loss: 1799.8330 - val_loss: 1721.4713
Epoch 6/7
6428/6428 [==============================] - 30s - loss: 1751.3002 - val_loss: 3055.6944
Epoch 7/7
6428/6428 [==============================] - 27s - loss: 2294.8134 - val_loss: 1682.8344
Saving the model..........................


# LeNet Model 
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.8.0.dylib locally
I tensorflow/stream_executor/dso_loader.cc:126] Couldn't open CUDA library libcudnn.5.dylib. LD_LIBRARY_PATH: /usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:/Developer/NVIDIA/CUDA-8.0/lib
I tensorflow/stream_executor/cuda/cuda_dnn.cc:3517] Unable to load cuDNN DSO
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.8.0.dylib locally
I tensorflow/stream_executor/dso_loader.cc:126] Couldn't open CUDA library libcuda.1.dylib. LD_LIBRARY_PATH: /usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:/Developer/NVIDIA/CUDA-8.0/lib
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.dylib locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.8.0.dylib locally
Total Samples: 8036
Readying the data set..........................

Getting the model..........................

Compiling the model..........................

Train on 6428 samples, validate on 1608 samples
Epoch 1/5
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
E tensorflow/stream_executor/cuda/cuda_driver.cc:509] failed call to cuInit: CUDA_ERROR_NOT_INITIALIZED
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: Not found: was unable to find libcuda.so DSO loaded into this program
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: Invalid argument: expected %d.%d or %d.%d.%d form for driver version; got ""
6428/6428 [==============================] - 210s - loss: 0.1543 - val_loss: 0.0168
Epoch 2/5
6428/6428 [==============================] - 196s - loss: 0.0140 - val_loss: 0.0156
Epoch 3/5
6428/6428 [==============================] - 198s - loss: 0.0120 - val_loss: 0.0120
Epoch 4/5
6428/6428 [==============================] - 165s - loss: 0.0102 - val_loss: 0.0124
Epoch 5/5
6400/6428 [============================>.] - ETA: 11s - loss: 0.0094  
6428/6428 [==============================] - 5483s - loss: 0.0094 - val_loss: 0.0123
Saving the model..........................


# After Augmenting the data
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.8.0.dylib locally
I tensorflow/stream_executor/dso_loader.cc:126] Couldn't open CUDA library libcudnn.5.dylib. LD_LIBRARY_PATH: /usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:/Developer/NVIDIA/CUDA-8.0/lib
I tensorflow/stream_executor/cuda/cuda_dnn.cc:3517] Unable to load cuDNN DSO
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.8.0.dylib locally
I tensorflow/stream_executor/dso_loader.cc:126] Couldn't open CUDA library libcuda.1.dylib. LD_LIBRARY_PATH: /usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:/Developer/NVIDIA/CUDA-8.0/lib
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.dylib locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.8.0.dylib locally
Total Samples: 8036
Readying the data set..........................

Getting the model..........................

Compiling the model..........................

Train on 12857 samples, validate on 3215 samples
Epoch 1/5
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
E tensorflow/stream_executor/cuda/cuda_driver.cc:509] failed call to cuInit: CUDA_ERROR_NOT_INITIALIZED
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: Not found: was unable to find libcuda.so DSO loaded into this program
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: Invalid argument: expected %d.%d or %d.%d.%d form for driver version; got ""
12857/12857 [==============================] - 338s - loss: 0.1094 - val_loss: 0.0122
Epoch 2/5
12857/12857 [==============================] - 336s - loss: 0.0110 - val_loss: 0.0111
Epoch 3/5
12857/12857 [==============================] - 365s - loss: 0.0102 - val_loss: 0.0111
Epoch 4/5
12857/12857 [==============================] - 398s - loss: 0.0096 - val_loss: 0.0113
Epoch 5/5
12857/12857 [==============================] - 344s - loss: 0.0091 - val_loss: 0.0113
Saving the model..........................


#After taking left and right camera along with already using center
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.8.0.dylib locally
I tensorflow/stream_executor/dso_loader.cc:126] Couldn't open CUDA library libcudnn.5.dylib. LD_LIBRARY_PATH: /usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:/Developer/NVIDIA/CUDA-8.0/lib
I tensorflow/stream_executor/cuda/cuda_dnn.cc:3517] Unable to load cuDNN DSO
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.8.0.dylib locally
I tensorflow/stream_executor/dso_loader.cc:126] Couldn't open CUDA library libcuda.1.dylib. LD_LIBRARY_PATH: /usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib:/Developer/NVIDIA/CUDA-8.0/lib
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.dylib locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.8.0.dylib locally
Total Samples: 8036
Readying the data set..........................

Image Samples: 48216
Getting the model..........................

Compiling the model..........................

Train on 38572 samples, validate on 9644 samples
Epoch 1/5
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
E tensorflow/stream_executor/cuda/cuda_driver.cc:509] failed call to cuInit: CUDA_ERROR_NOT_INITIALIZED
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: Not found: was unable to find libcuda.so DSO loaded into this program
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: Invalid argument: expected %d.%d or %d.%d.%d form for driver version; got ""
38572/38572 [==============================] - 18801s - loss: 0.0204 - val_loss: 0.0198
Epoch 2/5
38572/38572 [==============================] - 21876s - loss: 0.0166 - val_loss: 0.0205
Epoch 3/5
38572/38572 [==============================] - 1324s - loss: 0.0155 - val_loss: 0.0203
Epoch 4/5
38572/38572 [==============================] - 1383s - loss: 0.0147 - val_loss: 0.0194
Epoch 5/5
38572/38572 [==============================] - 1302s - loss: 0.0142 - val_loss: 0.0224
Saving the model..........................

Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x1307b1518>>
Traceback (most recent call last):
  File "/Users/kchandra/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 582, in __del__
AttributeError: 'NoneType' object has no attribute 'TF_DeleteStatus'


#Nvidia logs
Total Samples: 8036
Readying the data set..........................

Image Samples: 48216
Getting the model..........................

Compiling the model..........................

Train on 38572 samples, validate on 9644 samples
Epoch 1/3
E tensorflow/stream_executor/cuda/cuda_driver.cc:509] failed call to cuInit: CUDA_ERROR_UNKNOWN
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:145] kernel driver does not appear to be running on this host (ip-172-31-38-251): /proc/driver/nvidia/version does not exist
38572/38572 [==============================] - 395s - loss: 0.0175 - val_loss: 0.0172
Epoch 2/3
38572/38572 [==============================] - 394s - loss: 0.0149 - val_loss: 0.0192
Epoch 3/3
38572/38572 [==============================] - 394s - loss: 0.0135 - val_loss: 0.0191
Saving the model..........................

# Nvidia logs with generator
Reading the data folder: ../DataSets/carnd-behavioral-cloning-p3-data/data2
Image Samples: 6428
Getting the model..........................nvidia

Compiling the model..........................

Epoch 1/3
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
E tensorflow/stream_executor/cuda/cuda_driver.cc:509] failed call to cuInit: CUDA_ERROR_NOT_INITIALIZED
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: ZALANDO-25690
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: Not found: was unable to find libcuda.so DSO loaded into this program
I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: Invalid argument: expected %d.%d or %d.%d.%d form for driver version; got ""
6336/6428 [============================>.] - ETA: 1s - loss: 0.0255/Users/kchandra/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/keras/engine/training.py:1569: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.
  warnings.warn('Epoch comprised more than '
6528/6428 [==============================] - 136s - loss: 0.0251 - val_loss: 0.0288
Epoch 2/3
6528/6428 [==============================] - 157s - loss: 0.0212 - val_loss: 0.0177
Epoch 3/3
6528/6428 [==============================] - 158s - loss: 0.0193 - val_loss: 0.0190
dict_keys(['val_loss', 'loss'])
Saving the model..........................


