# video-human-counting
---

## Purpose
This repository mainly created for counting human benefit from video. 

## Method
Current crowd counting methods mainly focus on applying in single image. This repository proposed a CascadeCNN network, which utilizes multiple frames to refine the current frame dense map. Our Method can be divided into two kinds: Single Transformed CascadeCNN and Double transformed CascadeCNN.

## Preparing
### Prepare your environment(Recommand to use anaconda)
> ```shell
> conda create -n cascadecnn python=3.6
> pip install -r requirements.txt
> ```
### Data preparing
Since current datasets are basic based on single image, there are only Mall, UCSD, FDST are video based, so we train/test our method on Mall and FDST. You can download Mall[[link](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)] and FDST[[link](https://pan.baidu.com/share/init?surl=NNaJ1vtsxCPJUjDNhZ1sHA)(pwd:**sgt1**)]. We have offer code to process these two datasets. You can find in datasets/Mall/PrepareMall.m and /datasets/FDST/PrepareFDST.m. I suggest you to moditify the code to fit your environment, but the architecture of processed data should be like below:
- Folder Tree

    ```
    +-- Mall
    |   +-- train
    |       +--img
    |       +--den
    |   +-- test
    |       +--img
    |       +--den
    ```


## Train
In config.py you can change any parameter if you want. One important paramter is __C.STN. Change __C.STN to True so that the framework would adopt STN to process multiple frame dense maps. Create a folder named best_model, and put the single image trained model[[link](https://pan.baidu.com/s/1ld5s36CUFjcNDQMM2jlStw)](pwd: mupw) in it.
### Train
> ```shell
> python train.py
> ```

## Performance
Quantitative results(MAE/MSE):

|          Method                 | Mall         |
|---------------------------------|--------------|
| Kernel Ridge Regression         |3.51/18.10    |
| Ridge Regression                |3.59/19.00    |
| Gaussian Process Regression     |3.72/20.10    |
| Cumulative Attribute Regression |3.43/17.70    |
| COUNT Forest                    |2.50/10.00    |
| ConvLSTM                        |2.24/8.50     |
| Bidirectional ConvLSTM          |2.10/7.60     |
| LSTN                            |2.00/2.50     |
| DT-LCNN                         |2.03/2.60     |
| Single Transformed CascadeCNN   |1.7261/2.2233 |
| Double Transformed CascadeCNN   |1.7123/2.2000 |


|          Method                 | FDST         |
|---------------------------------|--------------|
| MCNN                            |3.77/4.88     |
| ConvLSTM                        |4.48/5.82     |
| LSTN                            |3.35/4.45     |
| Single Transformed CascadeCNN   |2.0902/2.7530 |
| Double Transformed CascadeCNN   |2.2550/2.8121 |


## Pretrained Model
We would upload  pretrained model to help you verify our method. You can find models of DTC[[link](https://pan.baidu.com/s/1B3LUv5Qh_3IAZE5OGFEqLA)](pwd: svek) trained on Mall.

