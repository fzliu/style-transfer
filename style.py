"""
style.py - An implementation of "A Neural Algorithm of Artistic Style"
by L. Gatys, A. Ecker, and M. Bethge. http://arxiv.org/abs/1508.06576.

authors: Frank Liu - frank@frankzliu.com
         Dylan Paiton - dpaiton@gmail.com
last modified: 09/15/2015

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
  - Expand arguments and options.
        - Grayscale output flag
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

# cudamat
try:
    import cudamat as cm
    cm.cublas_init()
    USE_CUDAMAT = True
except:
    USE_CUDAMAT = False


CAFFE_ROOT = os.path.abspath(os.path.join(os.path.dirname(caffe.__file__), "..", ".."))
MODEL_DIR = "models"

# weights for the individual models
VGG_WEIGHTS = {"content": {"conv4_2": 1},
               "style": {"conv1_1": 0.2,
                         "conv2_1": 0.2,
                         "conv3_1": 0.2,
                         "conv4_1": 0.2,
                         "conv5_1": 0.2}}
GOOGLENET_WEIGHTS = {"content": {"inception_3a/1x1": 0.5,
                                 "inception_3b/1x1": 0.5},
                     "style": {"conv1/7x7_s2": 0.125,
                               "conv2/3x3": 0.125,
                               "inception_3a/1x1": 0.125,
                               "inception_3b/1x1": 0.125,
                               "inception_4a/1x1": 0.125,
                               "inception_4b/1x1": 0.125,
                               "inception_4c/1x1": 0.125,
                               "inception_4d/1x1": 0.125}}
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
parser.add_argument("-m", "--model", default="googlenet", type=str, required=False, help="model to use")
parser.add_argument("-r", "--ratio", default="1e5", type=str, required=False, help="style-to-content ratio")
parser.add_argument("-i", "--max-iters", default=500, type=int, required=False, help="L-BFGS iterations")
parser.add_argument("-a", "--scale-output", default=1.0, type=float, required=False, help="output image scale")
parser.add_argument("-n", "--initialize", default="content", type=str, required=False, help="initialize gradient")
parser.add_argument("-d", "--debug", action="store_true", required=False, help="run in debug mode")
parser.add_argument("-o", "--output", default="output/result.jpg", required=False, help="output path")


def _compute_content_gradient(F, F_content, layer):
    """
        Computes content gradient from activation features.
    """

    # layer variables
    Fl = F[layer]

    # compute loss and gradient
    if USE_CUDAMAT:
        El = cm.empty(Fl.shape)
        Fl.subtract(F_content[layer], target=El)
        loss = El.euclid_norm()**2 / 2
        El.mult(Fl.greater_than(0))
        grad = El.asarray()
    else:
        El = Fl - F_content[layer]
        loss = (El**2).sum() / 2
        grad = El * (Fl>0)

    return loss, grad

def _compute_style_gradient(F, G_style, layer):
    """
        Computes style gradient from activation features.
    """

    # layer variables
    Fl = F[layer]
    (nl, ml) = Fl.shape
    c = nl**-2 * ml**-2

    # compute loss and gradient
    if USE_CUDAMAT:
        Gl = Fl.dot(Fl.T)
        El = cm.empty(Gl.shape)
        Gl.subtract(G_style[layer], target=El)
        loss = c/4 * El.euclid_norm()**2
        El = cm.dot(El, Fl, alpha=c)
        El.mult(Fl.greater_than(0))
        grad = El.asarray()
    else:
        Gl = sgemm(1.0, Fl, Fl.T)
        El = Gl - G_style[layer]
        loss = c/4 * (El**2).sum()
        grad = c * sgemm(1.0, El, Fl) * (Fl>0)

    return loss, grad

def _compute_representation(net, layers, data, do_gram=False):
    """
        Computes representation matrices for an image.
    """

    # copy activations to output from forward pass
    rep_mats = {}
    net.blobs["data"].data[0] = data

    #start = timeit.default_timer()
    net.forward(end=net.params.keys()[-1])
    #end = timeit.default_timer()
    #print("Single iteration took {0:.4f} seconds".format(end-start))

    for layer in layers:
        rep = net.blobs[layer].data[0].copy()

        # flatten filters before adding to output
        rep = rep.reshape(rep.shape[0], -1)
        if USE_CUDAMAT:
            rep = cm.CUDAMatrix(Fl, copy_on_host=False)

        # compute Gramian, if necessary
        if do_gram:
            rep = rep.dot(rep.T) if USE_CUDAMAT else sgemm(1.0, rep, rep.T)
        
        # update complete representation set
        rep_mats.update({layer: rep})

    return rep_mats

def style_optimizer(x, G_style, F_content, net, weights, ratio):
    """
        Style transfer optimization callback for scipy.optimize.minimize().
    """

    # initialize update params
    loss = 0
    layers = net.params.keys()
    net.blobs[layers[-1]].diff[:] = 0
    F = _compute_representation(net, G_style.keys()+F_content.keys(),
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
            loss += wl * l
            grad += wl * g.reshape(grad.shape)

        # style contribution
        if layer in weights["style"]:
            wl = weights["style"][layer]
            (l, g) = _compute_style_gradient(F, G_style, layer)
            loss += ratio * wl * l
            grad += ratio * wl * g.reshape(grad.shape)

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

    def __init__(self, model_name, model_dim):
        """
            Initialize the model used for style transfer.

            :param str model_name:
                Model to use.

            :param tuple model_dim:
                Model input dimensions.
        """

        base_path = os.path.join(MODEL_DIR, model_name)

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

        # load model
        self.load_model(model_file, pretrained_file, model_dim)
        self.weights = weights

    def load_model(self, model_file, pretrained_file, model_dim):
        """
            Loads specified model from caffe install (see caffe docs).

            :param str model_file:
                Path to model protobuf.

            :param str pretrained_file:
                Path to pretrained caffe model.
        """

        # load net
        net = caffe.Net(model_file, pretrained_file, caffe.TEST)
        net.blobs["data"].reshape(1, model_dim[2], *model_dim[0:2])

        # all models used are trained on imagenet data
        mean_path = os.path.join(CAFFE_ROOT, "python", "caffe", "imagenet", "ilsvrc_2012_mean.npy")
        transformer = caffe.io.Transformer({"data": net.blobs["data"].shape})
        transformer.set_mean("data", np.load(mean_path).mean(1).mean(1))
        transformer.set_channel_swap("data", (2,1,0))
        transformer.set_transpose("data", (2,0,1))
        transformer.set_raw_scale("data", 255)

        # add net parameters
        self.net = net
        self.transformer = transformer

    def save_generated(self, path):
        """
            Displays the generated image (net input).
        """

        # check output directory
        if os.path.dirname(path):
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

        # prettify the generated image and save it
        img = self.transformer.deprocess("data", self.net.blobs["data"].data)
        img = (255*img).astype(np.uint8)
        imsave(path, img)
    
    def initialize_input(self, initialize):
        """
            Creates an initial input (generated) image.
        """

        # specify dimensions and create grid in Fourier domain
        dims = tuple(self.net.blobs["data"].shape[2:]) + \
               (self.net.blobs["data"].shape[1], )
        grid = np.mgrid[0:dims[0], 0:dims[1]]
        beta = int(initialize)

        # create frequency representation for pink noise
        Sf = (grid[0] - (dims[0]-1)/2.0) ** 2 + \
             (grid[1] - (dims[1]-1)/2.0) ** 2
        Sf[np.where(Sf == 0)] = 1
        Sf = np.sqrt(Sf)
        Sf = np.dstack((Sf**beta,)*dims[2])

        # apply ifft to create pink noise and normalize
        ifft_kernel = np.cos(2*np.pi*np.random.randn(*dims)) + \
                      1j*np.sin(2*np.pi*np.random.randn(*dims))
        img_pink = np.abs(ifftn(Sf * ifft_kernel))
        img_pink -= img_pink.min()
        img_pink /= img_pink.max()

        # preprocess the pink noise image
        x0 = self.transformer.preprocess("data", img_pink)

        return x0

    def transfer_style(self, img_style, img_content, ratio=1e3, n_iter=500, initialize=None, debug=False):
        """
            Transfers the style of the artwork to the input image.

            :param numpy.ndarray img_style:
                A style image with the desired target style.

            :param numpy.ndarray img_content:
                A content image in 8-bit, RGB format.
        """

        # compute style representations
        style_layers = self.weights["style"].keys()
        style_data = self.transformer.preprocess("data", img_style)
        G_style = _compute_representation(self.net, style_layers, style_data, do_gram=True)

        # compute content representations
        content_layers = self.weights["content"].keys()
        content_data = self.transformer.preprocess("data", img_content)
        F_content = _compute_representation(self.net, content_layers, content_data)

        # generate initial net input
        # default = content image, see kaishengtai/neuralart
        if initialize == "content":
            net_in = content_data
        else:
            net_in = self.initialize_input(initialize)

        # compute data bounds
        pixel_min = -self.transformer.mean["data"][:,0,0]
        pixel_max = pixel_min + self.transformer.raw_scale["data"]
        data_bounds = [(pixel_min[0], pixel_max[0])]*(net_in.size/3) + \
                      [(pixel_min[1], pixel_max[1])]*(net_in.size/3) + \
                      [(pixel_min[2], pixel_max[2])]*(net_in.size/3)

        # perform optimization
        minfn_args = (G_style, F_content, self.net, self.weights, ratio)
        lbfgs_opts = {"maxiter": n_iter, "disp": debug}
        return minimize(style_optimizer, net_in.flatten(), 
                        args=minfn_args, method="L-BFGS-B", jac=True,
                        bounds=data_bounds, options=lbfgs_opts).nit


if __name__ == "__main__":
    args = parser.parse_args()

    if args.gpu_id == -1:
        logging.info("Running on CPU")
        caffe.set_mode_cpu()
        USE_CUDAMAT = False
    else:
        logging.info("Running on GPU {0}".format(args.gpu_id))
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    # load images
    img_style = caffe.io.load_image(args.style_img)
    img_content = caffe.io.load_image(args.content_img)
    
    # artistic style class
    out_shape = (int(args.scale_output * img_content.shape[0]),
                 int(args.scale_output * img_content.shape[1]),
                 img_content.shape[2])
    st = StyleTransfer(args.model, out_shape)

    # perform style transfer
    start = timeit.default_timer()
    n_iters = st.transfer_style(img_style, 
                                img_content, 
                                ratio=np.float(args.ratio),
                                n_iter=args.max_iters,
                                initialize=args.initialize,
                                debug=args.debug)
    end = timeit.default_timer()
    logging.info("Ran {0} iterations".format(n_iters))
    logging.info("Took {0:.0f} seconds".format(end-start))

    # DONE!
    st.save_generated(args.output)

    ## shutdown cudamat - segfaults?
    #if USE_CUDAMAT:
    #    cm.cublas_shutdown()
