# Semi-supervised Variational Autoencoder for Bearing Fault Diagnosis
This repository contains the codes for the course project of Deep Learning at Harbin Institute of Technology (Wangmeng Zuo, Wanxiang Che). I am trying out semi-supervised vae for bearing fault diagnosis. The dataset used for experiment is CWRU Bearing Dataset.

The vibration signals are min-max normalized and then sliced into segments of length 1024, and then FFT is used to transform the segments into frequency coefficients of length 1024 (so yeah, the left and right are symmetrical). The obtained frequency representation is reshaped into 32\*32. (This is probably not a good way to process the vibration signal. I may use 1D dilated convolution in the future.)

## 1. Using latent variables directly for classification (SVM as classifier)
Got pretty bad results as expected, and the accuracy declines dramatically with the decrease of training samples.

<img src="https://github.com/cyrilli/variational-autoencoder-bearing_supervised/blob/master/img/experiment1_result.png?raw=true" alt="图片名称" align=center />

## 2. Adding cross-entropy loss of labeled samples into original loss function
For each batch, half are labeled data, and half are unlabeled data. A softmax layer is added on top of latent variables, which gives logits of a batch. We can use the logits of the labeled samples and the corresponding labels to compute cross-entropy. This cross-entropy is added into the original loss function. The classification accuracy increased, and we can also see that accuracy slightly declines with the decrease of labeled samples.

<img src="https://github.com/cyrilli/variational-autoencoder-bearing_supervised/blob/master/img/experiment2_result.png?raw=true" alt="图片名称" align=center />

### 2.1 Input and reconstruction
<img src="https://github.com/cyrilli/variational-autoencoder-bearing_supervised/blob/master/img/experiment2_sample_reconstruction.png?raw=true" alt="图片名称" align=center />

### 2.2 Generation by drawing z from standard normal distribution
<img src="https://github.com/cyrilli/variational-autoencoder-bearing_supervised/blob/master/img/experiment2_generation.png?raw=true" alt="图片名称" align=center />

## 3. Future work
* Currently the encoder and decoder network are using 2D convolution. Consider use 1D dilated convolution instead.
