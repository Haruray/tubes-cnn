# tubes-cnn

## Penjelasan input gambar
Input gambar berupa matrix numpy dengan shape `(height, weight, channel)`. Namun, Convolution layer juga menangani bentuk numpy array `(height, weight)` saja, yang berarti juga menangani gambar grayscaled

## To do list
- Convolution Layer (almost done, kurang bias)
- Detector ✅
- Pooling ✅
- Flatten Layer 
- Dense Layer
- Data preprocess untuk prediksi gambar
- Proses Inference nya (Pembuatan Model)

## Info
kalau misal develop lewat VS Code dan dapet ini saat run pythonnya :
```
[ WARN:0@0.102] global loadsave.cpp:248 cv::findDecoder imread_('251.jpeg'): can't open/read file: check file path/integrity
```
Maka cara fix nya tinggal jalankan `cd src` di cmd nya

Referensi : 
- https://www.kaggle.com/code/milan400/cnn-from-scratch-numpy/notebook
- https://github.com/AlessandroSaviolo/CNN-from-Scratch
- https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
- https://www.pycodemates.com/2023/07/build-a-cnn-from-scratch-using-python.html#google_vignette
- https://www.deeplearningwizard.com/deep_learning/fromscratch/fromscratch_cnn/
- https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
