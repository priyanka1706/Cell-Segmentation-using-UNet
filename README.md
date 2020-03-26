# Cell Segmentation using UNet

The network is built from scratch based on the framework provided as part of the course CMPT 743 at Simon Fraser University.

The images to train on are rescaled to 572x572, with augmentations like rotating the image by 90 degrees, flipping it across the vertical axis and gamma correction with gamma=1.7.

While cross entropy is available in Pytorch, a hand crafted cross entropy function is used. It can be a little slow so for lesser training time the in built function can be used.

## Results
![alt text](https://github.com/priyanka1706/Cell-Segmentation-using-UNet/blob/master/Resullts.PNG)
