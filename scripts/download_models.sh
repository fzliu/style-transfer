#!/bin/bash

if [ "$#" == 0 ] || [ "$1" == "googlenet" ]; then
    curl "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" -o models/googlenet/bvlc_googlenet.caffemodel
fi
if [ "$#" == 0 ] || [ "$1" == "caffenet" ]; then
    curl "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel" -o models/caffenet/bvlc_reference_caffenet.caffemodel
fi
if [ "$#" == 0 ] || [ "$1" == "vgg16" ]; then
    curl "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel" -o models/vgg16/VGG_ILSVRC_16_layers.caffemodel
fi
if [ "$#" == 0 ] || [ "$1" == "vgg19" ]; then
    curl "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel" -o models/vgg19/VGG_ILSVRC_19_layers.caffemodel
fi
