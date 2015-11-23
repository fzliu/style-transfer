"""
styleapi.py - Style transfer API for optimized demo.
"""

# system imports
import argparse
import base64
import io
import logging
from multiprocessing import Pool
from threading import Lock
import time
import sys

# library imports
import caffe
import redis
from scipy.misc.pilutil import imshow
from skimage.transform import rescale
import redis
from rq import Connection, Queue, Worker
from rq import get_current_job

# local imports
from style import StyleTransfer

# argparse
parser = argparse.ArgumentParser(description="Start up style processes, which read from Redis.",
                                 usage="styleapi.py -n <n_gpu> -r <redis_url>")
parser.add_argument("-n", "--n-gpu", type=int, default=0, help="GPU to use")
parser.add_argument("-r", "--redis-url", type=str, required=True, help="redis DB URL0")

# style transfer instances, each should be backed by a lock
# this will be used only if style_api_init() is called
api_insts = {}

# single style transfer instance, for use with Redis DB
# this will only be used if style_worker_run() is called
style = None


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
    """

    global api_insts

    if n_workers == 0:
        n_workers = 1

    # assign a lock to each worker
    for i in range(n_workers):
        worker = StyleTransfer("googlenet", use_pbar=False, gpu_id=i)
        api_insts.update({Lock(): worker})

def style_api(img_style, img_content, callback=None):
    """
        Style transfer API.
    """

    global api_insts

    # style transfer arguments
    # can specify multiple sets of parameters
    all_args = [{"length": 512, "ratio": 1e5, "n_iter": 99, "callback": callback}]

    # acquire a worker (non-blocking)
    st_lock = None
    st_inst = None
    while st_lock is None:
        for lock, inst in api_insts.iteritems():

            # unblocking acquire
            if lock.acquire(False):
                st_lock = lock
                st_inst = inst
                break
            else:
                time.sleep(0.1)

    # start style transfer
    img_out = "content"
    for args in all_args:
        args["init"] = img_out
        st_inst.transfer_style(img_style, img_content, **args)
        img_out = st_inst.get_generated()
    st_lock.release()

    return img_out

def style_worker_run(worker_data):
    """
        Starts a style transfer worker, which reads image pairs from a Redis DB.
    """

    global style

    (redis_url, gpu_id) = worker_data

    # style worker
    style = StyleTransfer("googlenet", use_pbar=False, gpu_id=gpu_id)

    # start the worker
    conn = redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker(Queue("default"))
        worker.work()

def style_worker(img_style, img_content):
    """
        Performs style transfer.
    """

    global style

    # convert style image
    buf = io.BytesIO(base64.b64decode(img_style))
    img_style = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    buf.close()

    # convert content image
    buf = io.BytesIO(base64.b64decode(img_content))
    img_content = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
    buf.close()

    # initialize the job parameters
    job = get_current_job()
    job.meta["progress"] = 0
    job.meta["result"] = None
    job.save()

    # declare worker callback
    def callback(img_iter):
        job.meta["progress"] += 2
        job.meta["result"] = img_iter.tostring()
        job.save()

    # perform style transfer
    args = {"length": 512, "ratio": 1e5, "n_iter": 99, "callback": callback}
    style.transfer_style(img_style, img_content, **args)

def main(args):
    """
        Entry point.

        This function assumes that input and output images are backed by a Redis DB.
        For small-scale demos, import this module and call style_api() instead.
    """

    assert args.n_gpu >= 0, "must specify a non-negative number of GPUs"

    # number of GPUs to use
    n_gpu = gpu_count() if args.n_gpu == 0 else args.n_gpu
    if n_gpu == 0:
        sys.exit(0)

    # pool of style transfer processes
    pool = Pool(n_gpu)
    pool_params = zip((args.redis_url,)*2, range(0, n_gpu))
    pool.map(style_worker_run, pool_params)

    # shutdown
    pool.close()
    pool.join()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

