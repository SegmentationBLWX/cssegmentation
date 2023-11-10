# Get Started


## Prerequisites

In this section, we introduce some prerequisites for using CSSegmentation. 
If you are experienced with Python and PyTorch and have already installed them, just skip this part.

**1.Operation System**

CSSegmentation works on Linux, Windows and macOS. It requires Python 3.6+, CUDA 9.2+ and PyTorch 1.3+.

**2.Anaconda**

For Linux and Mac users, we strongly recommend you to download and install [Anaconda](https://docs.conda.io/en/latest/miniconda.html).
Then, you can create a conda environment for CSSegmentation and activate it,

```sh
conda create --name csseg python=3.8 -y
conda activate csseg
```

**3.WGET & Decompression Software**

If you want to utilize the provided scripts to prepare the datasets, it is necessary for you to install wget (for downloading datasets), 7z (for processing compressed packages) and tar (for processing compressed packages) in your operation system.
For windows users, the resources are listed as following,

- 7Z: [Download](https://sparanoid.com/lab/7z/download.html),
- RAR: [Download](https://www.win-rar.com/start.html?&L=0),
- WGET: [Download](http://downloads.sourceforge.net/gnuwin32/wget-1.11.4-1-setup.exe?spm=a2c6h.12873639.article-detail.7.3f825677H6sKF2&file=wget-1.11.4-1-setup.exe).

Besides, [Cmder](https://cmder.app/) are recommended to help the windows users execute the provided scripts successfully.

## Installation

**1.Clone CSSegmentation**

You can run the following commands to clone the cssegmentation repository,

```sh 
git clone https://github.com/SegmentationBLWX/cssegmentation.git
cd cssegmentation
```

**2.Install Requirements**

**2.1 Basic Requirements (Necessary)**

To set up the essential prerequisites for running CSSegmentation, execute the following commands,

```sh
pip install -r requirements.txt
```

This command will automatically install the following packages,

- `pillow`: set in requirements/io.txt,
- `pandas`: set in requirements/io.txt,
- `opencv-python`: set in requirements/io.txt,
- `inplace-abn`: set in requirements/nn.txt,
- `numpy`: set in requirements/science.txt,
- `scipy`: set in requirements/science.txt,
- `tqdm`: set in requirements/terminal.txt,
- `argparse`: set in requirements/terminal.txt,
- `cython`: set in requirements/misc.txt.

**2.2 Pytorch and Torchvision (Necessary)**

If you intend to utilize CSSegmentation, it is imperative to install PyTorch and torchvision. 
We recommend you to follow the [official instructions](https://pytorch.org/get-started/previous-versions/) to install them, *e.g.*,

```sh
# CUDA 11.0
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 10.2
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```