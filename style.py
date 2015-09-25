"""
style.py - An implementation of "A Neural Algorithm of Artistic Style"
by L. Gatys, A. Ecker, and M. Bethge. http://arxiv.org/abs/1508.06576.

authors: Frank Liu - frank@frankzliu.com
         Dylan Paiton - dpaiton@gmail.com
last modified: 09/25/2015

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
  - Add progress bar for minimize call
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
from scipy.fftpack import ifftn
from scipy.linalg.blas import sgemm
from scipy.misc import imsave
from scipy.optimize import minimize
from skimage.transform import rescale

try:
    import progressbar as pb
    gradIter = 0
    pbar = pb.ProgressBar()
    USE_PROGRESSBAR = True
except:
    USE_PROGRESSBAR = False 

CAFFE_ROOT = os.path.abspath(os.path.join(os.path.dirname(caffe.__file__), "..", ".."))
MEAN_PATH = os.path.join(CAFFE_ROOT, "python", "caffe", "imagenet", "ilsvrc_2012_mean.npy")
MODEL_DIR = "models"

# constants
INF = np.float32(np.inf)
STYLE_SCALE = 0.5

# weights for the individual models
VGG_WEIGHTS = {"content": {"conv4_2": 1},
               "style": {"conv1_1": 0.2,
                         "conv2_1": 0.2,
                         "conv3_1": 0.2,
                         "conv4_1": 0.2,
                         "conv5_1": 0.2}}
GOOGLENET_WEIGHTS = {"content": {"conv2/3x3": 0.005,
                                 "inception_3a/output": 0.995},
                     "style": {"conv1/7x7_s2": 0.1,
                               "conv2/3x3": 0.09,
                               "inception_3a/output": 0.09,
                               "inception_3b/output": 0.09,
                               "inception_4a/output": 0.09,
                               "inception_4b/output": 0.09,
                               "inception_4c/output": 0.09,
                               "inception_4d/output": 0.09,
                               "inception_4e/output": 0.09,
                               "inception_5a/output": 0.09,
                               "inception_5b/output": 0.09,}}
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
parser.add_argument("-n", "--num-iters", default=500, type=int, required=False, help="L-BFGS iterations")
parser.add_argument("-l", "--length", default=640, type=float, required=False, help="maximum image length")
parser.add_argument("-v", "--verbose", action="store_true", required=False, help="print minimization outputs")
parser.add_argument("-o", "--output", default="outputs/result.jpg", required=False, help="output path")


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

def _compute_reprs(net, layers_style, layers_content, net_in, scale_gram=1):
    """
        Computes representation matrices for an image.
    """

    # copy activations to output from forward pass
    (repr_s, repr_c) = ({}, {})
    net.blobs["data"].data[0] = net_in
    net.forward(end=net.params.keys()[-1])

    # loop through combined set of layers
    for layer in set(layers_style)|set(layers_content):
        F = net.blobs[layer].data[0].copy()
        F.shape = (F.shape[0], -1)
        repr_c[layer] = F
        if layer in layers_style:
            repr_s[layer] = sgemm(scale_gram, F, F.T)

    return repr_s, repr_c

def style_optfn(x, net, weights, G_style, F_content, ratio):
    """
        Style transfer optimization callback for scipy.optimize.minimize().
    """

    # initialize update params
    layers_all = net.params.keys()[::-1]
    layers_style = weights["style"].keys()
    layers_content = weights["content"].keys()
    net_in = x.reshape(net.blobs["data"].data.shape[1:])
    (G, F) = _compute_reprs(net, layers_style, layers_content, net_in)

    # backprop by layer
    loss = 0
    net.blobs[layers_all[-1]].diff[:] = 0
    for i, layer in enumerate(layers_all):
        next_layer = None if i == len(layers_all)-1 else layers_all[i+1]
        grad = net.blobs[layer].diff[0]

        # style contribution
        if layer in layers_style:
            wl = weights["style"][layer]
            (l, g) = _compute_style_grad(F, G, G_style, layer)
            loss += ratio * wl * l
            grad += ratio * wl * g.reshape(grad.shape)

        # content contribution
        if layer in layers_content:
            wl = weights["content"][layer]
            (l, g) = _compute_content_grad(F, F_content, layer)
            loss += wl * l
            grad += wl * g.reshape(grad.shape)

        # compute gradient
        net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]

    # format gradient for minimize() function
    grad = grad.flatten().astype(np.float64)

    return loss, grad

def progress_callback(Xi):
    global gradIter
    global pbar
    try:
        pbar.update(gradIter)
    except:
        pbar.finished = True
    gradIter += 1

class StyleTransfer(object):
    """
        Style transfer class.
    """

    def __init__(self, model_name):
        """
            Initialize the model used for style transfer.

            :param str model_name:
                Model to use.

            :param tuple scale_output:
                Output scale to use.
        """

        # googlenet
        if model_name == "googlenet":
            model_file = os.path.join(MODEL_DIR, model_name, "deploy.prototxt")
            pretrained_file = os.path.join(MODEL_DIR, model_name, "bvlc_googlenet.caffemodel")
            weights = GOOGLENET_WEIGHTS

        # vgg net
        elif model_name == "vgg":
            model_file = os.path.join(MODEL_DIR, model_name, "VGG_ILSVRC_19_layers_deploy.prototxt")
            pretrained_file = os.path.join(MODEL_DIR, model_name, "VGG_ILSVRC_19_layers.caffemodel")
            weights = VGG_WEIGHTS

        # default (caffenet)
        else:
            model_file = os.path.join(MODEL_DIR, model_name, "deploy.prototxt")
            pretrained_file = os.path.join(MODEL_DIR, model_name, "bvlc_reference_caffenet.caffemodel")
            weights = CAFFENET_WEIGHTS

        # load model and scale factors
        self.load_model(model_file, pretrained_file)
        self.weights = weights.copy()


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

        # load net (supressing output)
        out_orig = os.dup(2)
        null_fds = os.open(os.devnull, os.O_RDWR)
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

    def save_generated(self, path):
        """
            Saves the generated image (net input, after optimization).

            :param str path:
                Output path.
        """

        # check output directory
        if os.path.dirname(path):
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

        # prettify the generated image and save it
        img = self.transformer.deprocess("data", self.net.blobs["data"].data)
        img = (255*img).astype(np.uint8)
        imsave(path, img)
    
    def _rescale_net(self, img):
        """
            Rescales the network to fit a particular image.
        """

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

    def transfer_style(self, img_style, img_content, length=640,
                       ratio=1e5, n_iter=500, init=None, verbose=False):
        """
            Transfers the style of the artwork to the input image.

            :param numpy.ndarray img_style:
                A style image with the desired target style.

            :param numpy.ndarray img_content:
                A content image in 8-bit, RGB format.
        """

        # rescale the images
        scale = length / float(max(img_style.shape[:2]))
        img_style = rescale(img_style, STYLE_SCALE*scale, order=3)
        scale = length / float(max(img_content.shape[:2]))
        img_content = rescale(img_content, scale, order=3)

        # compute style representations
        self._rescale_net(img_style)
        layers = self.weights["style"].keys()
        net_in = self.transformer.preprocess("data", img_style)
        scale_gram = float(img_content.size)/img_style.size
        (G_style, _) = _compute_reprs(self.net, layers, [], net_in,
                                      scale_gram=scale_gram)

        # compute content representations
        self._rescale_net(img_content)
        layers = self.weights["content"].keys()
        net_in = self.transformer.preprocess("data", img_content)
        (_, F_content) = _compute_reprs(self.net, [], layers, net_in)

        # generate initial net input
        # default = content image, see kaishengtai/neuralart
        if init == "content":
            img0 = self.transformer.preprocess("data", img_content)
        else:
            img0 = self._make_noise_input(init)

        # compute data bounds
        data_min = -self.transformer.mean["data"][:,0,0]
        data_max = data_min + self.transformer.raw_scale["data"]
        data_bounds = [(data_min[0], data_max[0])]*(img0.size/3) + \
                      [(data_min[1], data_max[1])]*(img0.size/3) + \
                      [(data_min[2], data_max[2])]*(img0.size/3)

        # perform optimization
        minfn_args = (self.net, self.weights, G_style, F_content, ratio)
        lbfgs_opts = {"maxiter": n_iter, "disp": verbose}
        grad_method = "L-BFGS-B"

        if USE_PROGRESSBAR and not verbose:
            global pbar
            pbar.widgets = ['Progress: ', 
                pb.Percentage(), ' ', pb.Bar(marker=pb.AnimatedMarker()),
                ' ', pb.ETA()]
            pbar.maxval = n_iter
            pbar.start()
            return minimize(style_optfn, img0.flatten(), callback=progress_callback,
                            args=minfn_args, method=grad_method, jac=True, bounds=data_bounds,
                            options={"maxiter": n_iter, "disp": verbose}).nit
        else:
            if not verbose: print "Progress display requires the progressbar library."
            return minimize(style_optfn, img0.flatten(), args=minfn_args, 
                            method=grad_method, jac=True, bounds=data_bounds, 
                            options={"maxiter": n_iter, "disp": verbose}).nit

if __name__ == "__main__":
    args = parser.parse_args()

    # logging
    format = "%(filename)s:%(lineno)d  %(asctime)s -- %(message)s"
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(format=format, level=level)
    logging.info("Starting style transfer.")

    # set GPU/CPU mode
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
        logging.info("Running net on CPU.")
    else:
        caffe.set_device(args.gpu_id)
        logging.info("Running net on GPU {0}.".format(args.gpu_id))

    # load images
    img_style = caffe.io.load_image(args.style_img)
    img_content = caffe.io.load_image(args.content_img)
    logging.info("Successfully loaded images.")
    
    # artistic style class
    st = StyleTransfer(args.model.lower())
    logging.info("Successfully loaded model {0}.".format(args.model))

    # perform style transfer
    if not args.verbose: print "Creating art takes time..."
    start = timeit.default_timer()
    n_iters = st.transfer_style(img_style, img_content, length=args.length, 
                                init=args.init, ratio=np.float(args.ratio), 
                                n_iter=args.num_iters, verbose=args.verbose)
    end = timeit.default_timer()
    logging.info("Ran {0} iterations in {1:.0f}s.".format(n_iters, end-start))
    if USE_PROGRESSBAR and not args.verbose:
        pbar.finish()


    # DONE!
    st.save_generated(args.output)
    logging.info("Output saved to {0}.".format(args.output))
