'''
Function:
    Implementation of Transforms
Author:
    Zhenchao Jin
'''
import math
import torch
import random
import numbers
import warnings
import numpy as np
import collections
import torchvision.transforms.functional as F
from PIL import Image


'''Resize'''
class Resize(object):
    def __init__(self, output_size, image_interpolation='BILINEAR', seg_target_interpolation='NEAREST'):
        # assert
        assert isinstance(output_size, int) or \
               (isinstance(output_size, collections.Sequence) and len(output_size) == 2)
        assert hasattr(Image, image_interpolation) and hasattr(Image, seg_target_interpolation)
        # set attributes
        self.output_size = output_size
        self.image_interpolation = getattr(Image, image_interpolation)
        self.seg_target_interpolation = getattr(Image, seg_target_interpolation)
    '''call'''
    def __call__(self, data_meta):
        data_meta = self.resize('image', data_meta, self.output_size, self.image_interpolation)
        data_meta = self.resize('seg_target', data_meta, self.output_size, self.seg_target_interpolation)
        return data_meta
    '''resize'''
    @staticmethod
    def resize(key, data_meta, output_size, interpolation):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.resize(data_meta[key], output_size, interpolation)
        return data_meta


'''CenterCrop'''
class CenterCrop(object):
    def __init__(self, output_size):
        # set attributes
        if isinstance(output_size, numbers.Number):
            output_size = (int(output_size), int(output_size))
        self.output_size = output_size
    '''call'''
    def __call__(self, data_meta):
        data_meta = self.centercrop('image', data_meta, self.output_size)
        data_meta = self.centercrop('seg_target', data_meta, self.output_size)
        return data_meta
    '''centercrop'''
    @staticmethod
    def centercrop(key, data_meta, output_size):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.center_crop(data_meta[key], output_size)
        return data_meta


'''Pad'''
class Pad(object):
    def __init__(self, padding, image_fill=0, seg_target_fill=255, padding_mode='constant'):
        # assert
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(image_fill, (numbers.Number, str, tuple))
        assert isinstance(seg_target_fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence):
            assert len(padding) in [2, 4]
        # set attributes
        self.padding = padding
        self.image_fill = image_fill
        self.seg_target_fill = seg_target_fill
        self.padding_mode = padding_mode
    '''call'''
    def __call__(self, data_meta):
        data_meta = self.pad('image', data_meta, self.padding, self.image_fill, self.padding_mode)
        data_meta = self.pad('seg_target', data_meta, self.padding, self.seg_target_fill, self.padding_mode)
        return data_meta
    '''pad'''
    @staticmethod
    def pad(key, data_meta, padding, fill, padding_mode):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.pad(data_meta[key], padding, fill, padding_mode)
        return data_meta


'''Lambda'''
class Lambda(object):
    def __init__(self, lambd):
        # assert
        assert callable(lambd)
        # set attributes
        self.lambd = lambd
    '''call'''
    def __call__(self, data_meta):
        data_meta = self.lambd('image', data_meta, self.lambd)
        data_meta = self.lambd('seg_target', data_meta, self.lambd)
        return data_meta
    '''lambd'''
    @staticmethod
    def lambd(key, data_meta, lambd):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = lambd(data_meta[key])
        return data_meta


'''RandomRotation'''
class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        # assert
        assert isinstance(degrees, (numbers.Number, collections.Sequence))
        if isinstance(degrees, numbers.Number):
            assert degrees >= 0
            degrees = (-degrees, degrees)
        else:
            assert len(degrees) in [2]
        # set attributes
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
    '''call'''
    def __call__(self, data_meta):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        data_meta = self.randomrotate('image', data_meta, angle, self.resample, self.expand, self.center)
        data_meta = self.randomrotate('seg_target', data_meta, angle, self.resample, self.expand, self.center)
        return data_meta
    '''randomrotate'''
    @staticmethod
    def randomrotate(key, data_meta, angle, resample, expand, center):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.rotate(data_meta[key], angle, resample, expand, center)
        return data_meta


'''RandomHorizontalFlip'''
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        # assert
        assert isinstance(prob, numbers.Number)
        # set attributes
        self.prob = prob
    '''call'''
    def __call__(self, data_meta):
        if random.random() < self.prob:
            data_meta = self.hflip('image', data_meta)
            data_meta = self.hflip('seg_target', data_meta)
        return data_meta
    '''hflip'''
    @staticmethod
    def hflip(key, data_meta):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.hflip(data_meta[key])
        return data_meta


'''RandomVerticalFlip'''
class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        # assert
        assert isinstance(prob, numbers.Number)
        # set attributes
        self.prob = prob
    '''call'''
    def __call__(self, data_meta):
        if random.random() < self.prob:
            data_meta = self.vflip('image', data_meta)
            data_meta = self.vflip('seg_target', data_meta)
        return data_meta
    '''vflip'''
    @staticmethod
    def vflip(key, data_meta):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.vflip(data_meta[key])
        return data_meta


'''ToTensor'''
class ToTensor(object):
    def __init__(self):
        pass
    '''call'''
    def __call__(self, data_meta):
        if 'image' in data_meta:
            data_meta['image'] = F.to_tensor(data_meta['image'])
        if 'seg_target' in data_meta:
            data_meta['seg_target'] = torch.from_numpy(np.array(data_meta['seg_target'], dtype=np.uint8))
        return data_meta


'''Normalize'''
class Normalize(object):
    def __init__(self, mean, std):
        # set attributes
        self.mean = mean
        self.std = std
    '''call'''
    def __call__(self, data_meta):
        if 'image' in data_meta:
            data_meta['image'] = F.normalize(data_meta['image'], self.mean, self.std)
        return data_meta


'''RandomCrop'''
class RandomCrop(object):
    def __init__(self, output_size):
        # assert
        assert isinstance(output_size, (numbers.Number, collections.Sequence))
        # set attributes
        if isinstance(output_size, numbers.Number):
            output_size = (output_size, output_size)
        self.output_size = output_size
    '''call'''
    def __call__(self, data_meta):
        image_width, image_height = data_meta['image'].size
        output_height, output_width = self.output_size
        output_height = min(image_height, output_height)
        output_width = min(image_width, output_width)
        top, left, height, width = random.randint(0, image_height - output_height), random.randint(0, image_width - output_width), output_height, output_width
        data_meta = self.crop('image', data_meta, top, left, height, width)
        data_meta = self.crop('seg_target', data_meta, top, left, height, width)
        return data_meta
    '''crop'''
    @staticmethod
    def crop(key, data_meta, top, left, height, width):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.crop(data_meta[key], top, left, height, width)
        return data_meta


'''RandomResizedCrop'''
class RandomResizedCrop(object):
    def __init__(self, output_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), image_interpolation='BILINEAR', seg_target_interpolation='NEAREST'):
        # assert
        assert isinstance(output_size, int) or \
               (isinstance(output_size, collections.Sequence) and len(output_size) == 2)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert isinstance(scale, collections.Sequence) and len(scale) == 2
        assert scale[1] > scale[0]
        assert isinstance(ratio, collections.Sequence) and len(ratio) == 2
        assert ratio[1] > ratio[0]
        assert hasattr(Image, image_interpolation) and hasattr(Image, seg_target_interpolation)
        # set attributes
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio
        self.image_interpolation = getattr(Image, image_interpolation)
        self.seg_target_interpolation = getattr(Image, seg_target_interpolation)
    '''call'''
    def __call__(self, data_meta):
        image = data_meta['image']
        top, left, height, width = None, None, None, None
        area = image.size[0] * image.size[1]
        for _ in range(10):
            output_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            width = int(round(math.sqrt(output_area * aspect_ratio)))
            height = int(round(math.sqrt(output_area / aspect_ratio)))
            if width <= image.size[0] and height <= image.size[1]:
                top = random.randint(0, image.size[1] - height)
                left = random.randint(0, image.size[0] - width)
                break
        if top is None or left is None:
            in_ratio = image.size[0] / image.size[1]
            if (in_ratio < min(self.ratio)):
                width = image.size[0]
                height = int(round(width / min(self.ratio)))
            elif (in_ratio > max(self.ratio)):
                height = image.size[1]
                width = int(round(height * max(self.ratio)))
            else:
                width = image.size[0]
                height = image.size[1]
            top = (image.size[1] - height) // 2
            left = (image.size[0] - width) // 2
        data_meta = self.resizedcrop('image', data_meta, top, left, height, width, self.output_size, self.image_interpolation)
        data_meta = self.resizedcrop('seg_target', data_meta, top, left, height, width, self.output_size, self.seg_target_interpolation)
        return data_meta
    '''resizecrop'''
    @staticmethod
    def resizedcrop(key, data_meta, top, left, height, width, size, interpolation):
        if key in data_meta and data_meta[key] is not None:
            data_meta[key] = F.resized_crop(data_meta[key], top, left, height, width, size, interpolation)
        return data_meta


'''ColorJitter'''
class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None):
        # set attributes after checking
        self.brightness = self.check(brightness)
        self.contrast = self.check(contrast)
        self.saturation = self.check(saturation)
        self.hue = self.check(hue, center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
    '''call'''
    def __call__(self, data_meta):
        transforms = []
        # adjust brightness
        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda image: F.adjust_brightness(image, brightness_factor)))
        # adjust contrast
        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda image: F.adjust_contrast(image, contrast_factor)))
        # adjust saturation
        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda image: F.adjust_saturation(image, saturation_factor)))
        # adjust hue
        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda image: F.adjust_hue(image, hue_factor)))
        # random and perform
        random.shuffle(transforms)
        transforms = Compose(transforms)
        if 'image' in data_meta:
            data_meta['image'] = transforms(data_meta)
        # return
        return data_meta
    '''check'''
    def check(self, value, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if value is None: return value
        # assert
        assert isinstance(value, (numbers.Number, collections.Sequence))
        if isinstance(value, numbers.Number):
            assert value >= 0
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        else:
            assert bound[0] <= value[0] <= value[1] <= bound[1]
        # set
        return value


'''Compose'''
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    '''call'''
    def __call__(self, data_meta):
        for transform in self.transforms:
            data_meta = transform(data_meta)
        return data_meta