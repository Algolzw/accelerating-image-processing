import os
import sys
import cv2
import numpy as np
import pandas as pd
import random
from multiprocessing import Pool, Manager, set_start_method
import torch.multiprocessing as mp

import torch
from PIL import Image
from tqdm import tqdm

from utils import ProgressBar, random_degrade, usm_sharp

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def worker(sourcedir, outputdir, filepath):
    filename = filepath.split('/')[-1]
    # read high-quality image and normalize it to [0-1]
    image = cv2.imread(filepath) / 255.
    image_usm = usm_sharp(image)

    # generate low-quality image
    image_LQ = (random_degrade(image_usm) * 255).astype(np.uint8)
    lq_image_path = os.path.join(outputdir, filename)
    cv2.imwrite(lq_image_path, image_LQ)

    return "Processing {:s} ...".format(filename)

def generate_LQ_images(sourcedir, outputdir, num_repeat):
    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    filepaths = sorted(_get_paths_from_images(sourcedir))
    filepaths = filepaths * num_repeat # repeat 10000 times

    def update(arg):
        pbar.update(arg)
    pbar = ProgressBar(len(filepaths))

    pool = Pool(8)
    for filepath in filepaths:
        result = pool.apply_async(worker, args=(sourcedir, outputdir, filepath,), callback=update)

    pool.close()
    pool.join()
    
    print('Finished!!!')

if __name__ == "__main__":
    # set data dir
    sourcedir = "images"
    outputdir = "output"

    num_repeat = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    generate_LQ_images(sourcedir, outputdir, num_repeat)
