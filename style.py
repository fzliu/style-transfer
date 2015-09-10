"""
style.py - An implementation of "A Neural Algorithm of Artistic Style"
by L. Gatys, A. Ecker, and M. Bethge. http://arxiv.org/abs/1508.06576.

author: Frank Liu - frank@frankzliu.com
last modified: 09/09/2015

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Frank Liu (fzliu) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Frank Liu (fzliu) BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

TODOs:
  - Fix VGG Caffe model.
  - Move matrix operations to GPU.
  - Expand arguments and options.
"""

# system imports
import argparse
import os
import sys
import timeit

# library imports
import caffe
import numpy as np
from scipy.optimize import minimize
from skimage.io import imsave
from skimage.restoration import denoise_tv_bregman


CAFFE_ROOT = os.path.abspath(os.path.join(os.path.dirname(caffe.__file__), "..", ".."))
MODEL_DIR = "models"

# weights for the individual models
VGG_WEIGHTS = {"content": {"conv1_1": 1},
               "style": {"conv1_1": 0.2,
                         "conv2_1": 0.2,
                         "conv3_1": 0.2,
                         "conv4_1": 0.2,
                         "conv5_1": 0.2}}
GOOGLENET_WEIGHTS = {"content": {"conv1/7x7_s2": 1},
                     "style": {"conv1/7x7_s2": 0.125,
                               "conv2/3x3": 0.125,
                               "inception_3a/1x1": 0.125,
                               "inception_3b/1x1": 0.125,
                               "inception_4a/1x1": 0.125,
                               "inception_4b/1x1": 0.125,
                               "inception_4c/1x1": 0.125,
                               "inception_4d/1x1": 0.125}}
CAFFENET_WEIGHTS = {"content": {"conv1": 1},
                    "style": {"conv1": 0.2,
                              "conv2": 0.2,
                              "conv3": 0.2,
                              "conv4": 0.2,
                              "conv5": 0.2}}


# argparse
parser = argparse.ArgumentParser(description="Transfer the style of one image to another.",
                                 usage="style.py -s <style_image> -c <content_image> <new_height>")
parser.add_argument("-s", "--style-img", type=str, required=True, help="input style (art) image")
parser.add_argument("-c", "--content-img", type=str, required=True, help="input content image")
parser.add_argument("-m", "--model", default="googlenet", type=str, required=False, help="model to use")
parser.add_argument("-r", "--ratio", default=1.25e6, type=int, required=False, help="style-to-content ratio")
parser.add_argument("-i", "--max-iters", default=500, type=int, required=False, help="L-BFGS iterations")
parser.add_argument("-d", "--debug", action="store_true", required=False, help="run in debug mode")


def _compute_content_gradient(F, F_content, layer):
    """
        Computes content gradient from activation features.
    """

    # compute loss and gradient
    Fl = F[layer]
    Fl_delta = Fl-F_content[layer]
    loss = np.sum(Fl_delta**2)/2
    grad = Fl_delta# * (Fl>0)

    return loss, grad


def _compute_style_gradient(F, G_style, layer):
    """
        Computes style gradient from activation features.
    """

    # compute Gram matrix
    Fl = F[layer]
    Gl = Fl.dot(Fl.T)

    # compute loss and gradient
    (nl, ml) = Fl.shape
    c = 1.0/(nl**2*ml**2)
    Gl_delta = Gl - G_style[layer]
    loss = c/4*np.sum(Gl_delta**2)/4
    grad = c*Gl_delta.dot(Fl)# * (Fl>0)

    return loss, grad

def _compute_activations(net, layers, data):
    """
        Computes convolutional activation features for an image.
    """

    # copy activations to output from forward pass
    F = {}
    net.blobs["data"].data[0] = data
    net.forward(end=net.params.keys()[-1])
    for layer in layers:
        Fl = net.blobs[layer].data[0].copy()

        # flatten filters before adding to output
        Fl = Fl.reshape(Fl.shape[0], -1)
        F.update({layer: Fl})

    return F


def style_optimizer(x, G_style, F_content, net, weights, ratio):
    """
        Style transfer optimization callback for scipy.optimize.minimize().
    """

    # initialize update params
    loss = 0
    layers = net.params.keys()
    net.blobs[layers[-1]].diff[:] = 0
    F = _compute_activations(net, G_style.keys()+F_content.keys(),
                             x.reshape(net.blobs["data"].shape[1:]))

    # backprop by layer
    layers.reverse()
    for i, layer in enumerate(layers):
        next_layer = None if i == len(layers)-1 else layers[i+1]
        grad = net.blobs[layer].diff[0]

        # content contribution
        if layer in weights["content"]:
            wl = weights["content"][layer]
            (l, g) = _compute_content_gradient(F, F_content, layer)
            loss += wl*l
            grad += wl*g.reshape(grad.shape)

        # style contribution
        if layer in weights["style"]:
            wl = weights["style"][layer]
            (l, g) = _compute_style_gradient(F, G_style, layer)
            loss += ratio*wl*l
            grad += ratio*wl*g.reshape(grad.shape)

        # compute gradient
        net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]

    # format gradient for minimize() function
    grad = grad.flatten().astype(np.float64)

    return loss, grad


class StyleTransfer(object):
    """
        Style transfer class.
    """

    def __init__(self, model_name):
        """
            Initialize the model used for style transfer.

            :param str model_name:
                Model to use.
        """

        base_path = os.path.join(MODEL_DIR, model_name)

        # googlenet
        if model_name == "googlenet":
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "bvlc_googlenet.caffemodel")
            weights = GOOGLENET_WEIGHTS

        # caffenet
        elif model_name == "vgg":
            model_file = os.path.join(base_path, "VGG_ILSVRC_19_layers_deploy.prototxt")
            pretrained_file = os.path.join(base_path, "VGG_ILSVRC_19_layers.caffemodel")
            weights = VGG_WEIGHTS

        # default (caffenet)
        else:
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "bvlc_reference_caffenet.caffemodel")
            weights = CAFFENET_WEIGHTS

        # load model
        self.load_model(model_file, pretrained_file)
        self.weights = weights

    def load_model(self, model_file, pretrained_file):
        """
            Loads specified model from caffe install (see caffe docs).

            :param str model_file:
                Path to model protobuf.

            :param str pretrained_file:
                Path to pretrained caffe model.
        """

        # load net
        net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        net.blobs["data"].reshape(1, 3, 227, 227)

        # all models used are trained on imagenet data
        mean_path = os.path.join(CAFFE_ROOT, "python", "caffe", "imagenet", "ilsvrc_2012_mean.npy")
        transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
        transformer.set_mean("data", np.load(mean_path).mean(1).mean(1))
        transformer.set_channel_swap("data", (2,1,0))
        transformer.set_transpose("data", (2,0,1))
        transformer.set_raw_scale("data", 255)

        # add net parameters
        self.net = net
        self.transformer = transformer
        self.net_in = self.net.blobs["data"]

    def save_generated(self, path):
        """
            Displays the generated image (net input).
        """

        # prettify the generated image and show it
        img = self.transformer.deprocess("data", self.net_in.data)
        img = caffe.io.resize_image(img, self.orig_dims, interp_order=3)
        #img = denoise_tv_bregman(img, 20)
        imsave(path, img)

    def transfer_style(self, img_style, img_content, ratio=1e3, n_iter=500, debug=False):
        """
            Transfers the style of the artwork to the input image.

            :param numpy.ndarray img_style:
                A style image with the desired target style.

            :param numpy.ndarray img_content:
                A content image in 8-bit, RGB format.
        """

        self.orig_dims = img_content.shape

        # compute style representations
        style_layers = self.weights["style"].keys()
        style_data = self.transformer.preprocess("data", img_style)
        F_style = _compute_activations(self.net, style_layers, style_data)
        G_style = {l: F_style[l].dot(F_style[l].T) for l in F_style}

        # compute content representations
        content_layers = self.weights["content"].keys()
        content_data = self.transformer.preprocess("data", img_content)
        F_content = _compute_activations(self.net, content_layers, content_data)

        # initialize input with content image
        # from kaishengtai/neuralart
        data = self.transformer.preprocess("data", img_content)
        self.net_in.data[0] = data

        # compute data bounds
        pixel_min = -self.transformer.mean["data"][:,0,0]
        pixel_max = pixel_min+self.transformer.raw_scale["data"]
        data_bounds = [(pixel_min[0], pixel_max[0])]*(data.size/3) + \
                      [(pixel_min[1], pixel_max[1])]*(data.size/3) + \
                      [(pixel_min[2], pixel_max[2])]*(data.size/3)

        # perform optimization
        minfn_args = (G_style, F_content, self.net, self.weights, ratio)
        lbfgs_opts = {"maxiter": n_iter, "disp": debug}
        return minimize(style_optimizer, self.net_in.data.flatten(), 
                        args=minfn_args, method="L-BFGS-B", jac=True,
                        bounds=data_bounds, options=lbfgs_opts).nit


if __name__ == "__main__":
    caffe.set_mode_gpu()
    args = parser.parse_args()
    
    # artistic style class
    st = StyleTransfer(args.model)

    # style transfer
    start = timeit.default_timer()
    n_iters = st.transfer_style(caffe.io.load_image(args.style_img), 
                                caffe.io.load_image(args.content_img), 
                                ratio=args.ratio,
                                n_iter=args.max_iters,
                                debug=args.debug)
    end = timeit.default_timer()
    print("Ran {0} iterations".format(n_iters))
    print("Took {0:.0f} seconds".format(end-start))

    # DONE!
    st.save_generated("result.jpg")
