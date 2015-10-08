"""
demo.py - Optimized style transfer pipeline for interactive demo.
"""

# system imports
import argparse

# library imports
import caffe
import cv2
from skimage.transform import rescale

# local imports
from style import StyleTransfer

# argparse
parser = argparse.ArgumentParser(description="Run the optimized style transfer pipeline.",
                                 usage="demo.py -s <style_image> -c <content_image>")
parser.add_argument("-s", "--style-img", type=str, required=True, help="input style (art) image")
parser.add_argument("-c", "--content-img", type=str, required=True, help="input content image")

# style transfer
caffe.set_mode_cpu()
st = StyleTransfer("googlenet", use_pbar=True)


def st_api(img_style, img_content):
    """
        Style transfer API.
    """
    global st

    # run iterations
    all_args = [{"length": 360, "ratio": 4e2, "n_iter": 80},
                {"length": 480, "ratio": 4e3, "n_iter": 20},
                {"length": 640, "ratio": 4e4, "n_iter": 20}]
    img_out = "mixed"
    for args in all_args:
        args["init"] = img_out
        st.transfer_style(img_style, img_content, **args)
        img_out = st.get_generated()

    return img_out

def main(args):
    """
        Entry point.
    """

    # perform style transfer
    img_style = caffe.io.load_image(args.style_img)
    img_content = caffe.io.load_image(args.content_img)
    result = st_api(img_style, img_content)

    # show the image
    cv2.imshow("Art", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey()
    cv2.destroyWindow("Art")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
