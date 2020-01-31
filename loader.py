from cube_padding import cube_pad
import random
from PIL import Image
from torchvision.transforms import *
from torch.utils.data import Dataset, DataLoader
from functools import reduce
import sys
from pathlib import Path
import numpy as np
from imageio import imread
import torch


def samerandtrans(datadict, channel=3):
    from random import (randint, shuffle, sample, seed, choice)
    size = datadict['F'].size[0]
    cj = ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.01)
    rp = RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
    rc = RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
    rr = RandomRotation(90, resample=False, expand=False, center=None, fill=0)
    rv = RandomVerticalFlip(p=0.5)
    gs = Grayscale(channel)
    hf = RandomHorizontalFlip(p=0.5)
    vf = RandomVerticalFlip(p=0.5)

    flist = [cj, rp, rc, rr, rv, gs, hf, vf]
    randlist = sample(flist, randint(1, len(flist)))

    randseed = randint(0, sys.maxsize)
    randlist = choice([randlist, [lambda x: x]])
    transdict = {}
    for key, face in datadict.items():
        seed(randseed)
        transdict[key] = reduce(lambda arg, func: func(arg), randlist, face)
    #         for func in randlist:
    #             transdict[key] = func(face)
    return transdict


def getfromfold(foldname, stack=10):
    imgs = sorted(foldname.glob('*'), key=lambda x: int(x.stem))[:-1]
    N = len(imgs)
    if N < stack:
        imgs = imgs * ((stack // N) + 1)
    start = random.choice(range(len(imgs) - stack + 1))

    imgs = [*map(lambda x: x.as_posix(), imgs[start:start + stack])]
    flows = [*map(lambda x: (x.replace('images', 'flows/u'), x.replace('images', 'flows/v')), imgs)]
    return imgs, flows


class EgocentricData(Dataset):
    def __init__(self, imagepath='/data/keshav/360/finalEgok360/images/'):
        self.imlist = [*Path(imagepath).glob('*/*/*')]
        self.padsize = 10

    def __foldflows__(self, flows):
        flo = {'F': [], 'B': [], 'U': [], 'D': [], 'R': [], 'L': []}
        for floudict, flovdict in flows:
            for k in flo:
                u = ToTensor()(Grayscale()(floudict[k]))
                v = ToTensor()(Grayscale()(flovdict[k]))
                flo[k].append(torch.cat([u, v], 0))
        for k, f in flo.items():
            flo[k] = torch.stack(f, 1)
        return flo

    def __foldimages__(self, imgs):
        rgb = {'F': [], 'B': [], 'U': [], 'D': [], 'R': [], 'L': []}
        for imgdict in imgs:
            for k in rgb:
                rgb[k].append(ToTensor()(imgdict[k]))
        for k, i in rgb.items():
            rgb[k] = torch.stack(i, 1)
        return rgb

    def __loadwithtransform__(self, imlist):
        imgs, flows = getfromfold(imlist)
        try:
            imgs = [*map(lambda x: samerandtrans(cube_pad(imread(x), self.padsize)), imgs)]
            flows = [
                *map(lambda x: (cube_pad(imread(x[0]), self.padsize), cube_pad(imread(x[1]), self.padsize)), flows)]
        except Exception as e:
            print(str(e))

        return {'rgb': self.__foldimages__(imgs),
                'flo': self.__foldflows__(flows)}

    def __getitem__(self, idx):
        return self.__loadwithtransform__(self.imlist[idx])

    def __len__(self):
        return len(self.imlist)


def getdataloader():
    dataset = EgocentricData()
    dl = DataLoader(dataset=dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True)
    return dl