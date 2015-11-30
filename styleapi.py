"""
styleapi.py - Style transfer API for optimized demo.
"""

# system imports
import argparse
import base64
import io
import logging
import os
from threading import Lock
import time
import sys

# library imports
import caffe
import numpy as np
from PIL import Image
import redis
from rq import Connection, Queue, SimpleWorker
from rq import get_current_job
from skimage.transform import rescale
from skimage import img_as_ubyte

# local imports
from style import StyleTransfer


# style args
BASE_STYLE_ARGS = {"length": 512, "ratio": 1e5, "n_iter": 79}

# argparse
parser = argparse.ArgumentParser(description="Start up style processes, which read from Redis.",
                                 usage="styleapi.py -n <n_gpu> -r <redis_url>")
parser.add_argument("-r", "--redis-url", type=str, required=True, help="redis DB URL")

# style transfer objects, each backed by a lock, for use with style_api_init()
api_insts = {}

# @TODO: this is a hack to make the style transfer object visible during the job
# for a system with multiple GPUs, need to set CUDA_VISIBLE_DEVICES for each process
st_inst = StyleTransfer("googlenet", use_pbar=False, gpu_id=0)


def gpu_count():
    """
        Counts the number of CUDA-capable GPUs (Linux only).
    """

    # use nvidia-smi to count number of GPUs
    try:
        output = subprocess.check_output("nvidia-smi -L")
        return len(output.strip().split("\n"))
    except:
        return 0

def style_api_init(n_workers=1):
    """
        Initialize the style API backend.

        :param int n_workers:
            Number of style workers to initialize.
    """

    global api_insts

    # run in CPU mode
    if n_workers < 0:
        worker = StyleTransfer("googlenet", use_pbar=False, gpu_id=-1)
        api_insts.update({Lock(): worker})
        return

    # assign a lock to each worker
    for i in range(n_workers):
        worker = StyleTransfer("googlenet", use_pbar=False, gpu_id=i)
        api_insts.update({Lock(): worker})

def style_api(img_style, img_content, callback=None):
    """
        Style transfer API.

        :param numpy.ndarray img_style:
            A style image with the desired target style.

        :param numpy.ndarray img_content:
            A content image in floating point, RGB format.
    """

    global api_insts

    # style transfer arguments
    # can specify multiple sets of parameters
    all_args = [BASE_STYLE_ARGS.copy()]

    # acquire a worker (non-blocking)
    acq_lock = None
    acq_inst = None
    while acq_lock is None:
        for lock, inst in api_insts.iteritems():

            # unblocking acquire
            if lock.acquire(False):
                acq_lock = lock
                acq_inst = inst
                break
            else:
                time.sleep(0.1)

    # start style transfer
    img_out = "content"
    for args in all_args:
        args.update({"init": img_out, "callback": callback})
        acq_inst.transfer_style(img_style, img_content, **args)
        img_out = acq_inst.get_generated()
    acq_lock.release()

    return img_out

def image_to_base64(img):
    """
        Converts a floating point numpy array to base64-encoded JPEG.
    """

    # rescale image to max length of 512
    scale = 512.0/max(img.shape[:2])
    img = rescale(img, scale, order=3)

    # write the image to a buffer
    pimg = Image.fromarray(img_as_ubyte(img))
    buf = io.BytesIO()
    pimg.save(buf, format="JPEG")
    
    # base64 encode
    result = "data:image/jpg;base64,"
    result += base64.b64encode(buf.getvalue())

    return result

def style_worker(pimg_style_data, pimg_content_data):
    """
        Performs style transfer.
    """

    global st_inst

    # convert style and content image
    img_style = np.array(Image.frombytes(**pimg_style_data), dtype=np.float32)/255
    img_content = np.array(Image.frombytes(**pimg_content_data), dtype=np.float32)/255

    # initialize the job parameters
    job = get_current_job()
    job.meta["progress"] = 0.0
    job.meta["result"] = None
    job.save()

    # declare worker callback
    def callback(img_iter):
        job.meta["progress"] += 1.25
        job.meta["result"] = image_to_base64(img_iter)
        job.save()

    # perform style transfer
    args = BASE_STYLE_ARGS.copy()
    args.update({"init": "content", "callback": callback})
    st_inst.transfer_style(img_style, img_content, **args)
    img_result = st_inst.get_generated()
    result = image_to_base64(img_result)

    return result

def main(args):
    """
        Entry point.

        Executing this script will cause the process to listen for jobs from a Redis DB.
        This assumes that you have the right libraries (redis and rq) already installed.
        For small-scale demos, import this module and directly call style_api() instead.
    """

    # start it up
    conn = redis.from_url(args.redis_url)
    with Connection(conn):
        worker = SimpleWorker(Queue("default"))
        worker.work()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

