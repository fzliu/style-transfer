"""
style.py - An implementation of "A Neural Algorithm of Artistic Style"
by L. Gatys, A. Ecker, and M. Bethge. http://arxiv.org/abs/1508.06576.

authors: Frank Liu - frank@frankzliu.com
         Dylan Paiton - dpaiton@gmail.com
last modified: 12/04/2015

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
"""

# system imports
import argparse
import copy
import logging
import os
import sys
import tempfile
import timeit

# library imports
import caffe
from google.protobuf import text_format
import numpy as np
import progressbar as pb
from scipy.fftpack import ifftn
from scipy.linalg.blas import sgemm
from scipy.misc import imsave
from scipy.optimize import minimize
from skimage import img_as_ubyte
from skimage.transform import rescale

# local imports
from params import get_model_params


# logging
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"

# numeric constants
INF = np.float32(np.inf)
STYLE_SCALE = 1.2

# model-related constants
DATA = "data"
GRAM_SUFFIX = "/gram"
LOSS_SUFFIX = "/euclidean_loss"
LOSS_LABEL_SUFFIX = "/data"

# argparse
parser = argparse.ArgumentParser(description="Transfer the style of one image to another.",
                                 usage="style.py -s <style_image> -c <content_image>")
parser.add_argument("-s", "--style-img", type=str, required=True, help="input style (art) image")
parser.add_argument("-c", "--content-img", type=str, required=True, help="input content image")
parser.add_argument("-g", "--gpu-id", default=0, type=int, required=False, help="GPU device number")
parser.add_argument("-m", "--model", default="vgg16", type=str, required=False, help="model to use")
parser.add_argument("-i", "--init", default="content", type=str, required=False, help="initialization strategy")
parser.add_argument("-r", "--ratio", default="1e4", type=str, required=False, help="style-to-content ratio")
parser.add_argument("-n", "--num-iters", default=512, type=int, required=False, help="L-BFGS iterations")
parser.add_argument("-l", "--length", default=512, type=float, required=False, help="maximum image length")
parser.add_argument("-v", "--verbose", action="store_true", required=False, help="print minimization outputs")
parser.add_argument("-o", "--output", default=None, required=False, help="output path")


def style_optfn(x, net, weights, ratio):
    """
        Style transfer optimization callback for scipy.optimize.minimize().

        :param numpy.ndarray x:
            Flattened data array.

        :param caffe.Net net:
            Network to use to generate gradients.

        :param dict weights:
            Weights to use in the network.

        :param float ratio:
            Style-to-content ratio.
    """

    # prepare the input, then run the net
    net_in = x.reshape(net.blobs[DATA].data.shape[1:])
    net.blobs[DATA].data[0] = net_in
    net.forward()
    net.backward()

    # compute loss
    loss = 0.0
    for name, weight in weights["style"].iteritems():
        loss_name = name + GRAM_SUFFIX + LOSS_SUFFIX
        loss += net.blobs[loss_name].data * ratio * weight
    for name, weight in weights["content"].iteritems():
        loss_name = name + LOSS_SUFFIX
        loss += net.blobs[loss_name].data * weight

    # get gradients
    grad = net.blobs[DATA].diff[0]
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

            :param bool use_pbar:
                Use progressbar flag.
        """

        # load parameters and model
        (self.model_path,
         self.pretrained_path,
         self.mean_path,
         self.weights) = get_model_params(model_name)
        self._build_model()

        # add layers and other style transfer variables
        self.layers = []
        for layer in self.net.params.keys():
            if layer in self.weights["style"] or layer in self.weights["content"]:
                self.layers.append(layer)
        self.use_pbar = use_pbar

        # set the callback function
        if self.use_pbar:
            def callback(xk):
                self.grad_iter += 1
                try:
                    self.pbar.update(self.grad_iter)
                except:
                    self.pbar.finished = True
                if self._callback is not None:
                    net_in = xk.reshape(self.net.blobs[DATA].data.shape[1:])
                    self._callback(self.transformer.deprocess("data", net_in))
        else:
            def callback(xk):
                if self._callback is not None:
                    net_in = xk.reshape(self.net.blobs[DATA].data.shape[1:])
                    self._callback(self.transformer.deprocess("data", net_in))
        self.callback = callback

    def _make_gram_layer(self, base_name):
        """
            Creates a Gramian layer.
        """

        # make the layer, with gram params
        gram_name = base_name + GRAM_SUFFIX
        gram_params = caffe.io.caffe_pb2.GramianParameter(normalize_output=True)
        layer_params = {"bottom": [base_name], "top": [gram_name], 
                        "name": gram_name, "type": "Gramian",
                        "gramian_param": gram_params}

        return caffe.io.caffe_pb2.LayerParameter(**layer_params)

    def _make_loss_layer(self, base_name):
        """
            Creates a Euclidean loss layer.
        """

        # make the layer
        loss_name = base_name + LOSS_SUFFIX
        data_name = base_name + LOSS_LABEL_SUFFIX
        layer_params = {"bottom": [base_name, data_name], "top": [loss_name], 
                        "name": loss_name, "type": "EuclideanLoss"}

        return caffe.io.caffe_pb2.LayerParameter(**layer_params)

    def _add_data_params(self, model_def, input_name, data_shape):
        """
            Dynamically adds input data.
        """

        # create and add the input
        input_shape = caffe.io.caffe_pb2.BlobShape()
        input_shape.dim.extend(data_shape)
        model_def.input.append(input_name)
        model_def.input_shape.extend([input_shape])

    def _propogate_image(self, img):
        """
            Reshapes the network and propogates an image through it.
        """

        # get new dimensions and rescale net + transformer
        new_dims = (1, img.shape[2]) + img.shape[:2]
        self.net.blobs[DATA].reshape(*new_dims)
        self.transformer.inputs["data"] = new_dims

        # temporarily remove loss layers so reshape() succeeds
        # @TODO: will need to refactor this hack in the future
        added_layers = self.net.layers[-self._n_added_layers:]
        del self.net.layers[-self._n_added_layers:]
        self.net.reshape()

        # add the layers back, then reshape the inputs
        self.net.layers.extend(added_layers)
        for name in self.weights["content"]:
            data_name = name + LOSS_LABEL_SUFFIX
            data_shape = tuple(self.net.blobs[name].shape)
            self.net.blobs[data_name].reshape(*data_shape)
        self.net.reshape()

        # propogate the iamge
        net_in = self.transformer.preprocess(DATA, img)
        self.net.blobs[DATA].data[0] = net_in
        self.net.forward()

    def _make_noise_input(self, init):
        """
            Creates an initial input (generated) image.
        """

        # specify dimensions and create grid in Fourier domain
        dims = tuple(self.net.blobs[DATA].data.shape[2:]) + \
               (self.net.blobs[DATA].data.shape[1], )
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

    def _build_model(self):
        """
            Builds a style transfer model from a stock Caffe model.
        """

        # load dummy net (supressing stderr output)
        # this checks memory and correctness of net spec
        null_fds = os.open(os.devnull, os.O_RDWR)
        out_orig = os.dup(2)
        os.dup2(null_fds, 2)
        net = caffe.Net(self.model_path, caffe.TEST)
        
        # load the original Caffe definition
        model_def = caffe.io.caffe_pb2.NetParameter()
        with open(self.model_path) as fh:
            text_format.Merge(fh.read(), model_def)
        model_def.force_backward = True
        model_def.input_shape[0].dim[0] = 1

        # add layers to the style transfer model
        new_layers = []
        for layer in model_def.layer:
            name = layer.name

            # dynamically add layers for style
            if name in self.weights["style"]:
                gram_layer = self._make_gram_layer(name)
                loss_layer = self._make_loss_layer(gram_layer.name)
                data_shape = (1, ) + (net.blobs[name].shape[1], ) * 2
                self._add_data_params(model_def, loss_layer.bottom[1], data_shape)
                new_layers.append(gram_layer)
                new_layers.append(loss_layer)

            # dynamically add layers for content loss
            if name in self.weights["content"]:
                loss_layer = self._make_loss_layer(name)
                data_shape = (1, ) + tuple(net.blobs[name].shape[1:])
                self._add_data_params(model_def, loss_layer.bottom[1], data_shape)
                new_layers.append(loss_layer)

        # build new model file
        self._n_added_layers = len(new_layers)
        model_def.layer.extend(new_layers)
        model_file = tempfile.NamedTemporaryFile()
        model_file.write(str(model_def))
        model_file.flush()

        # reload net
        net = caffe.Net(model_file.name, self.pretrained_path, caffe.TEST)
        model_file.close()
        os.dup2(out_orig, 2)
        os.close(null_fds)

        # all models used are trained on imagenet data
        transformer = caffe.io.Transformer({"data": net.blobs[DATA].data.shape})
        transformer.set_mean("data", np.load(self.mean_path).mean(1).mean(1))
        transformer.set_channel_swap("data", (2,1,0))
        transformer.set_transpose("data", (2,0,1))
        transformer.set_raw_scale("data", 255)

        # add net parameters
        self.net = net
        self.transformer = transformer

    def get_generated(self):
        """
            Gets the generated image (net input, after optimization).
        """

        data = self.net.blobs[DATA].data
        img_out = self.transformer.deprocess("data", data)
        return img_out

    def transfer_style(self, img_style, img_content, length=512, ratio=1e5,
                       n_iter=512, init="-1", verbose=False, callback=None):
        """
            Transfers the style of the artwork to the input image.

            :param numpy.ndarray img_style:
                A style image with the desired target style.

            :param numpy.ndarray img_content:
                A content image in floating point, RGB format.

            :param int length:
                Maximum side length for the input images.

            :param float ratio:
                Style-to-content tradeoff ratio.

            :param int n_iter:
                Number of minimization iterations to run.

            :param str init:
                Input initialization strategy.

            :param function callback:
                A callback function, which takes images at iterations.
        """

        # assume that convnet input is square
        orig_dim = min(self.net.blobs[DATA].shape[2:])

        # rescale the images
        scale = max(length / float(max(img_style.shape[:2])),
                    orig_dim / float(min(img_style.shape[:2])))
        img_style = rescale(img_style, STYLE_SCALE*scale)
        scale = max(length / float(max(img_content.shape[:2])),
                    orig_dim / float(min(img_content.shape[:2])))
        img_content = rescale(img_content, scale)

        # compute and set style representations
        self._propogate_image(img_style)
        for name in self.weights["style"]:
            gram_name = name + GRAM_SUFFIX
            data_name = gram_name + LOSS_LABEL_SUFFIX
            self.net.blobs[data_name].data[:] = self.net.blobs[gram_name].data

        # compute and set content representations
        self._propogate_image(img_content)
        for name in self.weights["content"]:
            data_name = name + LOSS_LABEL_SUFFIX
            self.net.blobs[data_name].data[:] = self.net.blobs[name].data

        # generate initial net input
        # "content" = content image, see kaishengtai/neuralart
        if isinstance(init, np.ndarray):
            img0 = self.transformer.preprocess(DATA, init)
        elif init == "content":
            img0 = self.transformer.preprocess(DATA, img_content)
        elif init == "mixed":
            img0 = 0.95*self.transformer.preprocess(DATA, img_content) + \
                   0.05*self.transformer.preprocess(DATA, img_style)
        else:
            img0 = self._make_noise_input(init)

        # set the Euclidean loss diffs for style
        for name, weight in self.weights["style"].iteritems():
            loss_name = name + GRAM_SUFFIX + LOSS_SUFFIX
            self.net.blobs[loss_name].diff[...] = ratio * weight

        # set the Euclidean loss diffs for content
        for name, weight in self.weights["content"].iteritems():
            loss_name = name + LOSS_SUFFIX
            self.net.blobs[loss_name].diff[...] = weight

        # compute data bounds
        data_min = -self.transformer.mean[DATA][:,0,0]
        data_max = data_min + self.transformer.raw_scale[DATA]
        data_bounds = [(data_min[0], data_max[0])]*(img0.size/3) + \
                      [(data_min[1], data_max[1])]*(img0.size/3) + \
                      [(data_min[2], data_max[2])]*(img0.size/3)

        # optimization params
        minfn_args = {
            "args": (self.net, self.weights, ratio),
            "method": "L-BFGS-B", "jac": True, "bounds": data_bounds,
            "options": {"maxcor": 8, "maxiter": n_iter, "disp": verbose}
        }

        # optimize
        self._callback = callback
        minfn_args["callback"] = self.callback
        if self.use_pbar and not verbose:
            self._create_pbar(n_iter)
            self.pbar.start()
            res = minimize(style_optfn, img0.flatten(), **minfn_args).nit
            self.pbar.finish()
        else:
            res = minimize(style_optfn, img0.flatten(), **minfn_args).nit

        return res

def main(args):
    """
        Entry point.
    """

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
    use_pbar = not args.verbose
    st = StyleTransfer(args.model.lower(), use_pbar=use_pbar)
    logging.info("Successfully initialized model {0}.".format(args.model))

    # perform style transfer
    start = timeit.default_timer()
    n_iters = st.transfer_style(img_style, img_content, length=args.length, 
                                init=args.init, ratio=np.float(args.ratio), 
                                n_iter=args.num_iters, verbose=args.verbose)
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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

