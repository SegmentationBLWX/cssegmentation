## Introduction

<a href="https://github.com/fcdl94/MiB">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/cssegmentation/blob/main/csseg/modules/runners/mib.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2002.00718.pdf">MiB (CVPR'2020)</a></summary>

```latex
@inproceedings{cermelli2020modeling,
  title={Modeling the background for incremental learning in semantic segmentation},
  author={Cermelli, Fabio and Mancini, Massimiliano and Bulo, Samuel Rota and Ricci, Elisa and Caputo, Barbara},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9233--9242},
  year={2020}
}
```

</details>


## Results

#### PASCAL VOC

| Backbone    | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:         | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                      |
| R-101-D16   | 512x512    | 15-5-disjoint                       | 66.1%  | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/mib/mib_r101iabnd16_aspp_512x512_vocaug15-5_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_mib) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_mib/mib_r101iabnd16_aspp_512x512_vocaug15-5_disjoint.log)    |
| R-101-D16   | 512x512    | 15-5-overlapped                     | 70.2%  | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/mib/mib_r101iabnd16_aspp_512x512_vocaug15-5_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_mib) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_mib/mib_r101iabnd16_aspp_512x512_vocaug15-5_overlap.log)      |
| R-101-D16   | 512x512    | 15-1-disjoint                       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/mib/mib_r101iabnd16_aspp_512x512_vocaug15-1_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_mib) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_mib/mib_r101iabnd16_aspp_512x512_vocaug15-1_disjoint.log)    |
| R-101-D16   | 512x512    | 15-1-overlapped                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/mib/mib_r101iabnd16_aspp_512x512_vocaug15-1_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_mib) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_mib/mib_r101iabnd16_aspp_512x512_vocaug15-1_overlap.log)      |
| R-101-D16   | 512x512    | 10-1-disjoint                       | 9.9%   | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/mib/mib_r101iabnd16_aspp_512x512_vocaug10-1_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_mib) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_mib/mib_r101iabnd16_aspp_512x512_vocaug10-1_disjoint.log)    |
| R-101-D16   | 512x512    | 10-1-overlapped                     | 21.1%  | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/mib/mib_r101iabnd16_aspp_512x512_vocaug10-1_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_mib) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_mib/mib_r101iabnd16_aspp_512x512_vocaug10-1_overlap.log)      |

#### ADE20k

| Backbone   | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                       |
| :-:        | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                            |
| R-101-D16  | 512x512    | 100-50-disjoint                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 100-50-overlapped                   |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 100-10-disjoint                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 100-10-overlapped                   |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 100-5-disjoint                      |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 100-5-overlapped                    |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |


## More

You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1e9LlD6ITuLECstFcVPaCxQ with access code **55mq**