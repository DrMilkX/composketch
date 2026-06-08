from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
import requests
import random
import os
import pickle

from PIL import ImageEnhance

# previous modifications

def extend_side(im,side,amount):
    # majority_color = 255
    #majority_color = im.resize((1,1)).getpixel((0,0))
    # use random color from image as majority color
    majority_color = random.choice(im.getdata())
    if side == "left":
        new_im = PIL.Image.new(im.mode, (im.size[0]+amount, im.size[1]), majority_color)
        new_im.paste(im, (amount, 0))
    elif side == "right":
        new_im = PIL.Image.new(im.mode, (im.size[0]+amount, im.size[1]), majority_color)
        new_im.paste(im, (0, 0))
    elif side == "top":
        new_im = PIL.Image.new(im.mode, (im.size[0], im.size[1]+amount), majority_color)
        new_im.paste(im, (0, amount))
    elif side == "bottom":
        new_im = PIL.Image.new(im.mode, (im.size[0], im.size[1]+amount), majority_color)
        new_im.paste(im, (0, 0))
    return new_im

def random_crop(im):
    new_win_dim = (random.randint(im.size[0]//3,im.size[0]), random.randint(im.size[1]//3,im.size[1]))
    # apply random crop to random position
    x1 = random.randint(0, im.size[0] - new_win_dim[0])
    y1 = random.randint(0, im.size[1] - new_win_dim[1])
    x2 = x1 + new_win_dim[0]
    y2 = y1 + new_win_dim[1]
    
    return im.crop((x1, y1, x2, y2))

def change_contrast(im):
    # randomly change contrast by a factor of 0.5 to 1.5
    factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Contrast(im)
    return enhancer.enhance(factor)

def up_contrast(im):
    return ImageEnhance.Contrast(im).enhance(1.7)

def rotate_image(im):
    # randomly rotate image by 0 to 360 degrees
    angle = random.randint(0, 360)
    return im.rotate(angle)

def brightness(im):
    # randomly change brightness by a factor of 0.5 to 1.5
    factor = random.uniform(5, 7)
    enhancer = ImageEnhance.Brightness(im)
    return enhancer.enhance(factor)


def random_augment(im):
    # randomly apply one of the augmentations
    aug = random.choice([extend_side, random_crop, change_contrast, rotate_image, brightness])
    if aug == extend_side:
        side = random.choice(["left", "right", "top", "bottom"])
        amount = random.randint(10, 100)
        return extend_side(im,side,amount)
    else:
        return aug(im)
    

# reimport good images from pickle file
with open('peak-good_imgs.pkl', 'rb') as file:
    # Load the data from the file
    good_imgs = pickle.load(file)

print(good_imgs[0])


# Source - https://stackoverflow.com/a/61892265
# Posted by Karol Żak
# Retrieved 2026-06-08, License - CC BY-SA 4.0

def pixelate(im,level=8):
    org_size = im.size

    # scale it down
    im2 = im.resize(
        size=(org_size[0] // level, org_size[1] // level),
        resample=0)
    # and scale it up to get pixelate effect
    im2 = im2.resize(org_size, resample=0)
    return im2


# apply greyscale
def greyscale(im):
    return ImageOps.grayscale(im)


def mod_img(im):
    nimg = up_contrast(im)
    nimg = pixelate(nimg)
    nimg = greyscale(nimg)
    return nimg


# test on row of 8 images

if __name__ == '__main__':

    # pixelate(good_imgs[0][0]).show()

    sample_imgs = random.sample(good_imgs, min(8, len(good_imgs)))
    
    max_size = (200, 200)

    def fit_image(image):
        im = image.copy()
        im.thumbnail(max_size, Image.LANCZOS)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        return im

    originals = [fit_image(im[0]) for im in sample_imgs]
    pixelated = [fit_image(mod_img(im[0])) for im in sample_imgs]

    cols = len(originals)
    rows = 2
    grid = Image.new('RGB', (cols * max_size[0], rows * max_size[1]), (255, 255, 255))

    for idx, im in enumerate(originals):
        grid.paste(im, (idx * max_size[0], 0))

    for idx, im in enumerate(pixelated):
        grid.paste(im, (idx * max_size[0], max_size[1] + 20))

    grid.show()
