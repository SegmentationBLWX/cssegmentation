<div align="center">
  <img src="./docs/logo.png" width="600"/>
</div>
<br />

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://cssegmentation.readthedocs.io/en/latest/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cssegmentation)](https://pypi.org/project/cssegmentation/)
[![PyPI](https://img.shields.io/pypi/v/cssegmentation)](https://pypi.org/project/cssegmentation)
[![license](https://img.shields.io/github/license/SegmentationBLWX/cssegmentation.svg)](https://github.com/SegmentationBLWX/cssegmentation/blob/master/LICENSE)
[![PyPI - Downloads](https://pepy.tech/badge/cssegmentation)](https://pypi.org/project/cssegmentation/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/cssegmentation?style=flat-square)](https://pypi.org/project/cssegmentation/)
[![issue resolution](https://isitmaintained.com/badge/resolution/SegmentationBLWX/cssegmentation.svg)](https://github.com/SegmentationBLWX/cssegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/SegmentationBLWX/cssegmentation.svg)](https://github.com/SegmentationBLWX/cssegmentation/issues)

Documents: https://cssegmentation.readthedocs.io/en/latest/


## Introduction

CSSegmentation: An Open Source Continual Semantic Segmentation Toolbox Based on PyTorch.
You can star this repository to keep track of the project if it's helpful for you, thank you for your support.


## Major Features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various continual semantic segmentation methods.

- **Modular Design**

  We decompose the continual semantic segmentation framework into different components and one can easily construct a customized continual semantic segmentation framework by combining different modules.
 
- **Support of Multiple Methods Out of Box**

  The toolbox directly supports popular and contemporary continual semantic segmentation frameworks, *e.g.*, PLOP, RCIL, etc.
 
- **High Performance**

  The segmentation performance is better than or comparable to other codebases.
  

## Benchmark and Model Zoo

#### Supported Encoder

- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)

#### Supported Decoder

- [Deeplabv3](https://arxiv.org/pdf/1706.05587.pdf)

#### Supported Runner

- [PLOP]()
- [RCIL]()

#### Supported Datasets

- [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)


## Citation

If you use this framework in your research, please cite this project:

```
@misc{csseg2023,
    author = {Zhenchao Jin},
    title = {CSSegmentation: An Open Source Continual Semantic Segmentation Toolbox Based on PyTorch},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/SegmentationBLWX/cssegmentation}},
}
```