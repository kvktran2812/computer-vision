This repository is mean for me to study computer vision by implementing and integrate well-known architecture like Alex Net, Residual Net, Inception Net, etc. I will implement those legacy models and try to understand how it's run    

There are a fews things that I change different from those papers:
- I train on CIFAR-10 and CIFAR-100 mostly because the datasets are much smaller, easier to train on personal computer while the legacy models train on ImageNet with millions of images and 1000 classes.
- Some models have slight modification to make more sense of the resolution of CIFAR datasets.

Here are full implementation:
- [AlexNet](https://github.com/kvktran2812/computer-vision/blob/main/alex_net.ipynb)
- [ResNet](https://github.com/kvktran2812/computer-vision/blob/main/res_net.ipynb)
