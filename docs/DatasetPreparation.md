# Dataset Preparation


## Dataset Notes

**1.Download Source**

For easier io reading, some supported datasets have been pre-processed like creating the train.txt/val.txt/test.txt used to record the corresponding imageids.
So, it is recommended to adopt the provided script (*i.e.*, `scripts/prepare_datasets.sh`) to download the supported datasets or download the supported datasets from the provided network disk link rather than official website.


## Supported Datasets

**1.ADE20k**

- Official Website: [click](https://groups.csail.mit.edu/vision/datasets/ADE20K/),
- Baidu Disk: [click](https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw) (access code: fn1i),
- Script Command: `bash scripts/prepare_datasets.sh ade20k`.

**2.PASCAL VOC**

- Official Website: [click](http://host.robots.ox.ac.uk/pascal/VOC/),
- Baidu Disk: [click](https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw) (access code: fn1i),
- Script Command: `bash scripts/prepare_datasets.sh pascalvoc`.

**3.CityScapes**

- Official Website: [click](https://www.cityscapes-dataset.com/),
- Baidu Disk: [click](https://pan.baidu.com/s/1TZbgxPnY0Als6LoiV80Xrw) (access code: fn1i),
- Script Command: `bash scripts/prepare_datasets.sh cityscapes`.