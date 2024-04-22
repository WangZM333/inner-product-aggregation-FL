# 内积聚合策略

基于一个FedAVG算法框架实现的内积聚合算法

Some codes refers to https://github.com/Zing22/tf-fed-demo， https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/input_data.py and  https://github.com/persistforever/cifar10-tensorflow/blob/master/src/dataloader/cifar10.py

### environment
##### Tensorflow-version

1.python3.7.6

2.tensorflow1.13.1

##### PyTorch-version

1.python3.7.6

2.pytorch1.4.0

both of them run on GPU

### prepare data sets

You are supposed to prepare the data set by yourself. MNIST can be downloaded on http://yann.lecun.com/exdb/mnist/, and CIFAR-10 can be downloaded on http://www.cs.toronto.edu/~kriz/cifar.html. These data sets should be put into /data/MNIST and /data/CIFAR-10 when the download is finished.

### usage

Run FedAvg

```asp
python server.py -nc 100 -cf 0.1 -E 1 -B 10 -mn MLR  -ncomm 20 -iid 1 -lr 0.01 -vf 20 -g 0
```
Run inner product
```asp
python server_inner_product.py -nc 100 -cf 0.1 -E 1 -B 10  -mn MLR  -ncomm 20 -iid 1 -lr 0.01 -vf 20 -g 0
```

which means there are 100 clients,  we randomly select 10 in each communicating round.  The data set are allocated in IID way.  The epoch and batch size are set to 1 and 10. The learning rate is 0.01, we validate the codes every 20 rounds during the training, training stops after 20 rounds. There are three models to do experiments: mlr mnist_2nn mnist_cnn and cifar_cnn, and we choose mnist_cnn in this command. Notice the data set path when run the code of pytorch-version(you can take the source code out of the 'use_pytorch' folder). 



[1] Mcmahan H B , Moore E , Ramage D , et al. Communication-Efficient Learning of Deep Networks from Decentralized Data[J]. 2016.