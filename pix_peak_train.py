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
        new_im = Image.new(im.mode, (im.size[0]+amount, im.size[1]), majority_color)
        new_im.paste(im, (amount, 0))
    elif side == "right":
        new_im = Image.new(im.mode, (im.size[0]+amount, im.size[1]), majority_color)
        new_im.paste(im, (0, 0))
    elif side == "top":
        new_im = Image.new(im.mode, (im.size[0], im.size[1]+amount), majority_color)
        new_im.paste(im, (0, amount))
    elif side == "bottom":
        new_im = Image.new(im.mode, (im.size[0], im.size[1]+amount), majority_color)
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


def mass_modification(im):
    nimg = up_contrast(im)
    nimg = pixelate(nimg)
    nimg = greyscale(nimg)
    return nimg

# ==== MODEL TRAINING === #


"""
Image Quality Classifier
Trains a binary classifier: 1 = good image, 0 = bad image (cropped/padded).

Directory structure expected:
    dataset/
        good/   ← images labeled 1
        bad/    ← images labeled 0
"""

import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = "peakmodel_dataset/modded"          # root folder with good/ and bad/ subfolders
IMG_SIZE    = 224                # all images resized to this
BATCH_SIZE  = 16
EPOCHS      = 15
LR          = 1e-4
VAL_SPLIT   = 0.2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH   = "peak_model_modded.pt"
# ─────────────────────────────────────────────────────────────────────────────


def get_transforms():
    """
    Augment heavily — the dataset is varied and bad images differ only in
    geometric distortions, so spatial augmentation helps generalisation.
    """
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),     # handle random input sizes
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet stats
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes=2):
    """EfficientNet-B0 pretrained on ImageNet, head replaced for binary task."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class InMemoryDataset(Dataset):
    """Dataset built from in-memory lists of [PIL.Image, metadata] pairs."""
    def __init__(self, samples, labels, transform=None):
        self.samples   = samples    # list of PIL Images
        self.labels    = labels     # list of int labels (1=good, 0=bad)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx].convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_datasets(good_imgs_mod, bad_imgs_mod, train_tf, val_tf):
    """Build train/val datasets from in-memory good and bad image lists."""
    images = [item[0] for item in good_imgs_mod] + [item[0] for item in bad_imgs_mod]
    labels = [1] * len(good_imgs_mod) + [0] * len(bad_imgs_mod)
    print(f"Total samples: {len(images)} ({len(good_imgs_mod)} good, {len(bad_imgs_mod)} bad)")

    # Shuffle before splitting
    combined = list(zip(images, labels))
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(combined), generator=rng).tolist()
    images = [combined[i][0] for i in indices]
    labels = [combined[i][1] for i in indices]

    n_val   = int(len(images) * VAL_SPLIT)
    n_train = len(images) - n_val

    train_ds = InMemoryDataset(images[:n_train], labels[:n_train], transform=train_tf)
    val_ds   = InMemoryDataset(images[n_train:], labels[n_train:], transform=val_tf)

    return train_ds, val_ds


def train(good_imgs_mod, bad_imgs_mod):
    train_tf, val_tf = get_transforms()
    train_ds, val_ds = load_datasets(good_imgs_mod, bad_imgs_mod, train_tf, val_tf)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model (val acc {val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {SAVE_PATH}")


# ── Inference helper ──────────────────────────────────────────────────────────

def predict(image_path: str, model_path: str = SAVE_PATH) -> dict:
    """
    Predict whether a single image is good (1) or bad (0).
    Works on images of any size.
    """
    from PIL import Image

    _, val_tf = get_transforms()
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img    = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    label = int(probs.argmax().item())
    return {
        "label":       label,
        "prediction":  "good" if label == 1 else "bad",
        "confidence":  round(probs[label].item(), 4),
        "prob_good":   round(probs[1].item(), 4),
        "prob_bad":    round(probs[0].item(), 4),
    }



if __name__ == "__main__":

    # reimport good images from pickle file
    with open('peak-good_imgs.pkl', 'rb') as file:
        # Load the data from the file
        good_imgs = pickle.load(file)


    # Apply mass modification to the good images
    good_imgs_mod = [[mass_modification(im[0]), im[1]] for im in good_imgs]

    # bad dataset - extend a side or crop it poorly
    bad_imgs = []
    shuffled_imgs = random.sample(good_imgs, k=len(good_imgs))
    for im in shuffled_imgs:
        # randomly augment the image
        bad_imgs.append([random_augment(im[0]),im[1]])

    bad_imgs_mod = [[mass_modification(im[0]), im[1]] for im in bad_imgs]


    # export to dataset folder
    train(good_imgs_mod, bad_imgs_mod)

    # test on the test set
    imgs = {}
    for s in os.listdir("imgs/test-imgs/shapes"):
        imgs[s] = Image.open("imgs/test-imgs/shapes/"+s).convert("RGB")
    for s, im in imgs.items():
        print(f"{s}: {predict(None, mass_modification(im))}")

