# video-human-counting
---

## Purpose
This repository mainly created for counting human(without duplication) in video. 

## Acknowledge
This code is modified from the original opensource crowd counting framework code https://github.com/gjy3035/C-3-Framework

## Method
Current crowd counting methods mainly focus on applying in single image. This repository proposed a CascadeCNN network, which utilizes multiple frames to refine the current frame dense map. Our Method can be divided into two kinds: Native CascadeCNN and STN transformed CascadeCNN.

## Data Preparing
Since current datasets are basic based on single image, there are only Mall, UCSD, FDST are video based, so we train/test our method on Mall and FDST. You can download Mall[[link](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html)] and FDST[[LINK](https://pan.baidu.com/share/init?surl=NNaJ1vtsxCPJUjDNhZ1sHA)(pwd:**sgt1**)]. We have offer code to process these two datasets. Meanwhile, we offer processed data[[link]].

## Train & Test
In config.py you can change any parameter if you want. One important paramter is __C.STN. Change __C.STN to True so that the framework would adopt STN to process multiple frame dense maps.
### Train
> ```shell
> python train.py
> ```
### Test
> ```shell
> python test.py
> ```

## Performance


## Pretrained Model
We would upload some pretrained models to help you verify our method. You can find models of CascadeCNN[[link]()]/CascadeCNN+STN[[link]()] trained on Mall and th corresponding models[[link]()][[link]()] trained on FDST.

