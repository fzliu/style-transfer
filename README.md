# style-transfer

## Introduction

This repository contains a pyCaffe-based implementation of "A Neural Algorithm of Artistic Style" by L. Gatys, A. Ecker, and M. Bethge, which presents a method for transferring the artistic style of one input image onto another. You can read the paper here: http://arxiv.org/abs/1508.06576. 

Neural net operations are handled by Caffe, while loss minimization and other miscellaneous matrix operations are performed using numpy and scipy. L-BFGS is used for minimization.

## Requirements

 - Python >= 2.7
 - Caffe == latest

## Download

To run the code, you must have Caffe installed and the appropriate Python bindings in your `PYTHONPATH` environment variable. Detailed installation instructions for Caffe can be found [here](http://caffe.berkeleyvision.org/installation.html).

All of the necessary code is contained in the file `style.py`. You can try it on your own style and content image by running the following command:

```
python style.py -s <style_image> -c <content_image> -m <model_name>
```

The protobufs which come with the vanilla Caffe install aren't quite compatible with this code - working ones have already been added to this repository as a result of this. To get the pretrained models, simply run:

```
bash scripts/download_models.sh
```

This will grab the convnet models from the links provided in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). You may also specify the exact model you'd like to download by running:

```
bash scripts/download_models.sh <model_name>
```

Here, `<model_name>` must be one of `caffenet`, `googlenet`, or `vgg`.

## Sample

Original image: [San Francisco](https://www.flickr.com/photos/anhgemus-photography/15377047497) by Anh Dinh, licenced via Creative Commons. The result was generated in less than 100 BFGS iterations with random initialization of the image.

<p align="center">
<img src="https://raw.githubusercontent.com/fzliu/style-transfer/master/images/starry_night.jpg" width="50%"/>
</p>
<p>
<img src="https://raw.githubusercontent.com/fzliu/style-transfer/master/images/san_francisco.jpg" width="40%"/>
<img src="https://raw.githubusercontent.com/fzliu/style-transfer/master/images/starry_sanfran.jpg" width="40%"/>
</p>

More examples coming soon.
