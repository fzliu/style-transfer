"""
style.py - An implementation of "A Neural Algorithm of Artistic Style"
by L. Gatys, A. Ecker, and M. Bethge. http://arxiv.org/abs/1508.06576.

authors: Frank Liu - frank@frankzliu.com
         Dylan Paiton - dpaiton@gmail.com
last modified: 09/28/2015

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
  - Verify model_WEIGHTS is working as expected
"""

# system imports
import argparse
import logging
import os
import sys
import timeit

# library imports
import caffe
import numpy as np
import progressbar as pb
from scipy.fftpack import ifftn
from scipy.linalg.blas import sgemm
from scipy.misc import imsave
from scipy.optimize import minimize
from skimage import img_as_ubyte
from skimage.transform import rescale


# logging
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"

# paths for Caffe and models
CAFFE_ROOT = os.path.abspath(os.path.join(os.path.dirname(caffe.__file__), "..", ".."))
MEAN_PATH = os.path.join(CAFFE_ROOT, "python", "caffe", "imagenet", "ilsvrc_2012_mean.npy")

# numeric constants
INF = np.float32(np.inf)
STYLE_SCALE = 1.0

# weights for the individual models
VGG_WEIGHTS = {"content": {"conv4_2": 1},
               "style": {"conv1_1": 0.2,
                         "conv2_1": 0.2,
                         "conv3_1": 0.2,
                         "conv4_1": 0.2,
                         "conv5_1": 0.2}}
GOOGLENET_WEIGHTS = {"content": {"inception_3a/output": 1},
                     "style": {"conv1/7x7_s2": 0.2,
                               "conv2/3x3": 0.2,
                               "inception_3a/output": 0.2,
                               "inception_4a/output": 0.2,
                               "inception_5a/output": 0.2}}
CAFFENET_WEIGHTS = {"content": {"conv3": 1},
                    "style": {"conv1": 0.2,
                              "conv2": 0.2,
                              "conv3": 0.2,
                              "conv4": 0.2,
                              "conv5": 0.2}}

# argparse
parser = argparse.ArgumentParser(description="Transfer the style of one image to another.",
                                 usage="style.py -s <style_image> -c <content_image>")
parser.add_argument("-s", "--style-img", type=str, required=True, help="input style (art) image")
parser.add_argument("-c", "--content-img", type=str, required=True, help="input content image")
parser.add_argument("-g", "--gpu-id", default=-1, type=int, required=False, help="GPU device number")
parser.add_argument("-m", "--model", default="vgg", type=str, required=False, help="model to use")
parser.add_argument("-i", "--init", default="content", type=str, required=False, help="initialization strategy")
parser.add_argument("-r", "--ratio", default="1e5", type=str, required=False, help="style-to-content ratio")
parser.add_argument("-n", "--num-iters", default=512, type=int, required=False, help="L-BFGS iterations")
parser.add_argument("-l", "--length", default=512, type=float, required=False, help="maximum image length")
parser.add_argument("-v", "--verbose", action="store_true", required=False, help="print minimization outputs")
parser.add_argument("-o", "--output", default=None, required=False, help="output path")


def _compute_style_grad(F, G, G_style, layer):
    """
        Computes style gradient and loss from activation features.
    """

    # compute loss and gradient
    (Fl, Gl) = (F[layer], G[layer])
    c = Fl.shape[0]**-2 * Fl.shape[1]**-2
    El = Gl - G_style[layer]
    loss = c/4 * (El**2).sum()
    grad = c * sgemm(1.0, El, Fl) * (Fl>0)

    return loss, grad

def _compute_content_grad(F, F_content, layer):
    """
        Computes content gradient and loss from activation features.
    """

    # compute loss and gradient
    Fl = F[layer]
    El = Fl - F_content[layer]
    loss = (El**2).sum() / 2
    grad = El * (Fl>0)

    return loss, grad

def _compute_reprs(net_in, net, layers_style, layers_content, gram_scale=1):
    """
        Computes representation matrices for an image.
    """

    # copy data and forward pass
    (repr_s, repr_c) = ({}, {})
    net.blobs["data"].data[0] = net_in
    net.forward(end=net.params.keys()[-1])

    # loop through combined set of layers
    for layer in set(layers_style)|set(layers_content):
        F = net.blobs[layer].data[0].copy()
        F.shape = (F.shape[0], -1)
        repr_c[layer] = F
        if layer in layers_style:
            repr_s[layer] = sgemm(gram_scale, F, F.T)

    return repr_s, repr_c

def style_optfn(x, net, weights, layers, reprs, ratio):
    """
        Style transfer optimization callback for scipy.optimize.minimize().
    """

    # update params
    layers_style = weights["style"].keys()
    layers_content = weights["content"].keys()
    net_in = x.reshape(net.blobs["data"].data.shape[1:])

    # compute representations
    (G_style, F_content) = reprs
    (G, F) = _compute_reprs(net_in, net, layers_style, layers_content)

    # backprop by layer
    loss = 0
    net.blobs[layers[-1]].diff[:] = 0
    for i, layer in enumerate(reversed(layers)):
        next_layer = None if i == len(layers)-1 else layers[-i-2]
        grad = net.blobs[layer].diff[0]

        # style contribution
        if layer in layers_style:
            wl = weights["style"][layer]
            (l, g) = _compute_style_grad(F, G, G_style, layer)
            loss += wl * l
            grad += wl * g.reshape(grad.shape)

        # content contribution
        if layer in layers_content:
            wl = weights["content"][layer]
            (l, g) = _compute_content_grad(F, F_content, layer)
            loss += wl * l / ratio
            grad += wl * g.reshape(grad.shape) / ratio

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

    def __init__(self, model_name, use_pbar=True):
        """
            Initialize the model used for style transfer.

            :param str model_name:
                Model to use.

            :param tuple scale_output:
                Output scale to use.
        """

        base_path = os.path.join("models", model_name)

        # googlenet
        if model_name == "googlenet":
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "bvlc_googlenet.caffemodel")
            weights = GOOGLENET_WEIGHTS

        # vgg net
        elif model_name == "vgg":
            model_file = os.path.join(base_path, "VGG_ILSVRC_19_layers_deploy.prototxt")
            pretrained_file = os.path.join(base_path, "VGG_ILSVRC_19_layers.caffemodel")
            weights = VGG_WEIGHTS

        # default (caffenet)
        else:
            model_file = os.path.join(base_path, "deploy.prototxt")
            pretrained_file = os.path.join(base_path, "bvlc_reference_caffenet.caffemodel")
            weights = CAFFENET_WEIGHTS

        # add model and weights
        self.load_model(model_file, pretrained_file)
        self.weights = weights.copy()
        self.layers = []
        for layer in self.net.params.keys():
            if layer in self.weights["style"] or layer in self.weights["content"]:
                self.layers.append(layer)
        self.use_pbar = use_pbar

        # create progress bar callback
        if self.use_pbar:
            def pbar_cbfn(xk):
                self.grad_iter += 1
                try:
                    self.pbar.update(self.grad_iter)
                except:
                    self.pbar.finished = True
            self.pbar_cbfn = pbar_cbfn

    def load_model(self, model_file, pretrained_file):
        """
            Loads specified model from caffe install (see caffe docs).

            :param str model_file:
                Path to model protobuf.

            :param str pretrained_file:
                Path to pretrained caffe model.
        """

        assert(os.path.isfile(model_file))
        assert(os.path.isfile(pretrained_file))

        # load net (supressing stderr output)
        null_fds = os.open(os.devnull, os.O_RDWR)
        out_orig = os.dup(2)
        os.dup2(null_fds, 2)
        net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        os.dup2(out_orig, 2)
        os.close(null_fds)

        # all models used are trained on imagenet data
        transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
        transformer.set_mean("data", np.load(MEAN_PATH).mean(1).mean(1))
        transformer.set_channel_swap("data", (2,1,0))
        transformer.set_transpose("data", (2,0,1))
        transformer.set_raw_scale("data", 255)

        # add net parameters
        self.net = net
        self.transformer = transformer

    def get_generated(self):
        """
            Saves the generated image (net input, after optimization).

            :param str path:
                Output path.
        """

        data = self.net.blobs["data"].data
        img_out = self.transformer.deprocess("data", data)
        return img_out
    
    def _rescale_net(self, img):
        """
            Rescales the network to fit a particular image.
        """

        # get new dimensions and rescale net + transformer
        new_dims = (1, img.shape[2]) + img.shape[:2]
        self.net.blobs["data"].reshape(*new_dims)
        self.transformer.inputs["data"] = new_dims

    def _make_noise_input(self, init):
        """
            Creates an initial input (generated) image.
        """

        # specify dimensions and create grid in Fourier domain
        dims = tuple(self.net.blobs["data"].data.shape[2:]) + \
               (self.net.blobs["data"].data.shape[1], )
        grid = np.mgrid[0:dims[0], 0:dims[1]]

        # create frequency representation for pink noise
        Sf = (grid[0] - (dims[0]-1)/2.0) ** 2 + \
             (grid[1] - (dims[1]-1)/2.0) ** 2
        Sf[np.where(Sf == 0)] = 1
        Sf = np.sqrt(Sf)
        Sf = np.dstack((Sf**int(init),)*dims[2])

        # apply ifft to create pink noise and normalize
        ifft_kernel = np.cos(2*np.pi*np.random.randn(*dims)) + \
                      1j*np.sin(2*np.pi*np.random.randn(*dims))
        img_noise = np.abs(ifftn(Sf * ifft_kernel))
        img_noise -= img_noise.min()
        img_noise /= img_noise.max()

        # preprocess the pink noise image
        x0 = self.transformer.preprocess("data", img_noise)

        return x0

    def _create_pbar(self, max_iter):
        """
            Creates a progress bar.
        """

        self.grad_iter = 0
        self.pbar = pb.ProgressBar()
        self.pbar.widgets = ["Optimizing: ", pb.Percentage(), 
                             " ", pb.Bar(marker=pb.AnimatedMarker()),
                             " ", pb.ETA()]
        self.pbar.maxval = max_iter

    def transfer_style(self, img_style, img_content, length=512,
                       ratio=1e5, n_iter=500, init="-1", verbose=False):
        """
            Transfers the style of the artwork to the input image.

            :param numpy.ndarray img_style:
                A style image with the desired target style.

            :param numpy.ndarray img_content:
                A content image in floating point, RGB format.
        """

        # rescale the images
        scale = length / float(max(img_style.shape[:2]))
        img_style = rescale(img_style, STYLE_SCALE*scale)
        scale = length / float(max(img_content.shape[:2]))
        img_content = rescale(img_content, scale)

        # compute style representations
        self._rescale_net(img_style)
        layers = self.weights["style"].keys()
        net_in = self.transformer.preprocess("data", img_style)
        gram_scale = float(img_content.size)/img_style.size
        G_style = _compute_reprs(net_in, self.net, layers, [],
                                 gram_scale=1)[0]

        # compute content representations
        self._rescale_net(img_content)
        layers = self.weights["content"].keys()
        net_in = self.transformer.preprocess("data", img_content)
        F_content = _compute_reprs(net_in, self.net, [], layers)[1]

        # generate initial net input
        # "content" = content image, see kaishengtai/neuralart
        if init == "content":
            img0 = self.transformer.preprocess("data", img_content)
        elif isinstance(init, np.ndarray):
            img0 = self.transformer.preprocess("data", init)
        else:
            img0 = self._make_noise_input(init)

        # compute data bounds
        data_min = -self.transformer.mean["data"][:,0,0]
        data_max = data_min + self.transformer.raw_scale["data"]
        data_bounds = [(data_min[0], data_max[0])]*(img0.size/3) + \
                      [(data_min[1], data_max[1])]*(img0.size/3) + \
                      [(data_min[2], data_max[2])]*(img0.size/3)

        # optimization params
        grad_method = "L-BFGS-B"
        reprs = (G_style, F_content)
        minfn_args = {
            "args": (self.net, self.weights, self.layers, reprs, ratio),
            "method": grad_method, "jac": True, "bounds": data_bounds,
            "options": {"maxiter": n_iter, "disp": verbose}
        }

        # optimize
        if self.use_pbar and not verbose:
            self._create_pbar(n_iter)
            minfn_args["callback"] = self.pbar_cbfn
            self.pbar.start()
            res = minimize(style_optfn, img0.flatten(), **minfn_args).nit
            self.pbar.finish()
        else:
            res = minimize(style_optfn, img0.flatten(), **minfn_args).nit

        return res

if __name__ == "__main__":
    args = parser.parse_args()

    # logging
    level = logging.INFO if args.verbose else logging.DEBUG
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logging.info("Starting style transfer.")

    # set GPU/CPU mode
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
        logging.info("Running net on CPU.")
    else:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
        logging.info("Running net on GPU {0}.".format(args.gpu_id))

    # load images
    img_style = caffe.io.load_image(args.style_img)
    img_content = caffe.io.load_image(args.content_img)
    logging.info("Successfully loaded images.")
    
    # artistic style class
    st = StyleTransfer(args.model.lower(), use_pbar=True)
    logging.info("Successfully loaded model {0}.".format(args.model))

    # perform style transfer
    img_out = None
    for i in range(2, -1, -1):
        logging.info("Minimization pass {0} of 3.".format(3-i))
        length = args.length // 2**i
        init = args.init if img_out is None else img_out
        ratio = np.float(args.ratio) / 64**i
        n_iter = args.num_iters // 4**(2-i)
        start = timeit.default_timer()
        n_iters = st.transfer_style(img_style, img_content, length=length,
                                    init=init, ratio=ratio, n_iter=n_iter,
                                    verbose=args.verbose)
        end = timeit.default_timer()
        logging.info("Ran {0} iterations in {1:.0f}s.".format(n_iters, end-start))
        img_out = st.get_generated()

    # output path
    if args.output is not None:
        out_path = args.output
    else:
        out_path_fmt = (os.path.splitext(os.path.split(args.content_img)[1])[0], 
                        os.path.splitext(os.path.split(args.style_img)[1])[0], 
                        args.model, args.init, args.ratio, args.num_iters)
        out_path = "outputs/{0}-{1}-{2}-{3}-{4}-{5}.jpg".format(*out_path_fmt)

    # DONE!
    imsave(out_path, img_as_ubyte(img_out))
    logging.info("Output saved to {0}.".format(out_path))
