"""
params.py - Model parameters for style transfer.
"""
import os

# weights for the individual models, by output blob
# assume that corresponding layers' top blob matches its name
VGG19_WEIGHTS = {"content": {"conv4_2": 1},
                 "style": {"conv1_1": 0.2,
                           "conv2_1": 0.2,
                           "conv3_1": 0.2,
                           "conv4_1": 0.2,
                           "conv5_1": 0.2}}
VGG16_WEIGHTS = {"content": {"conv4_2": 1},
                 "style": {"conv1_1": 0.2,
                           "conv2_1": 0.2,
                           "conv3_1": 0.2,
                           "conv4_1": 0.2,
                           "conv5_1": 0.2}}
GOOGLENET_WEIGHTS = {"content": {"inception_3b/output": 1},
                     "style": {"conv1/7x7_s2": 0.2,
                               "conv2/3x3": 0.2,
                               "inception_3a/output": 0.2,
                               "inception_4a/output": 0.2,
                               "inception_5a/output": 0.2}}
CAFFENET_WEIGHTS = {"content": {"conv4": 1},
                    "style": {"conv1": 0.2,
                              "conv2": 0.2,
                              "conv3": 0.2,
                              "conv4": 0.2,
                              "conv5": 0.2}}

def get_model_params(model_name):
    """
        Get model parameters for style transfer.
    """

    style_path = os.path.abspath(os.path.split(__file__)[0])
    base_path = os.path.join(style_path, "models", model_name)

    # vgg19
    if model_name == "vgg19":
        model_path = os.path.join(base_path, "VGG_ILSVRC_19_layers_deploy.prototxt")
        pretrained_path = os.path.join(base_path, "VGG_ILSVRC_19_layers.caffemodel")
        mean_path = os.path.join(base_path, "ilsvrc_2012_mean.npy")
        weights = VGG19_WEIGHTS

    # vgg16
    elif model_name == "vgg16":
        model_path = os.path.join(base_path, "VGG_ILSVRC_16_layers_deploy.prototxt")
        pretrained_path = os.path.join(base_path, "VGG_ILSVRC_16_layers.caffemodel")
        mean_path = os.path.join(base_path, "ilsvrc_2012_mean.npy")
        weights = VGG16_WEIGHTS

    # googlenet
    elif model_name == "googlenet":
        model_path = os.path.join(base_path, "deploy.prototxt")
        pretrained_path = os.path.join(base_path, "googlenet_style.caffemodel")
        mean_path = os.path.join(base_path, "ilsvrc_2012_mean.npy")
        weights = GOOGLENET_WEIGHTS

    # caffenet
    elif model_name == "caffenet":
        model_path = os.path.join(base_path, "deploy.prototxt")
        pretrained_path = os.path.join(base_path, "bvlc_reference_caffenet.caffemodel")
        mean_path = os.path.join(base_path, "ilsvrc_2012_mean.npy")
        weights = CAFFENET_WEIGHTS

    else:
        assert False, "model not available"

    return (model_path, pretrained_path, mean_path, weights)
