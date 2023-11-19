## Introduction

<a href="https://github.com/arthurdouillard/CVPR2021_PLOP">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/cssegmentation/blob/main/csseg/modules/runners/plop.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2011.11390.pdf">PLOP (CVPR'2021)</a></summary>

```latex
@inproceedings{douillard2021plop,
  title={Plop: Learning without forgetting for continual semantic segmentation},
  author={Douillard, Arthur and Chen, Yifu and Dapogny, Arnaud and Cord, Matthieu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4040--4050},
  year={2021}
}
```

</details>


## Results

#### PASCAL VOC

| Backbone   | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                     |
| :-:        | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                          |
| R-101-D16  | 512x512    | 15-5-disjoint                       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_vocaug15-5_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_vocaug15-5_disjoint.log)   |
| R-101-D16  | 512x512    | 15-5-overlapped                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_vocaug15-5_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_vocaug15-5_overlap.log)     |
| R-101-D16  | 512x512    | 15-1-disjoint                       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_vocaug15-1_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_vocaug15-1_disjoint.log)   |
| R-101-D16  | 512x512    | 15-1-overlapped                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_vocaug15-1_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_vocaug15-1_overlap.log)     |
| R-101-D16  | 512x512    | 10-1-disjoint                       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_vocaug10-1_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_vocaug10-1_disjoint.log)   |
| R-101-D16  | 512x512    | 10-1-overlapped                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_vocaug10-1_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_vocaug10-1_overlap.log)     |

#### ADE20k

| Backbone   | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                        |
| :-:        | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                             |
| R-101-D16  | 512x512    | 100-50-disjoint                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_ade20k100-50_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_ade20k100-50_disjoint.log)  |
| R-101-D16  | 512x512    | 100-50-overlapped                   |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_ade20k100-50_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_ade20k100-50_overlap.log)    |
| R-101-D16  | 512x512    | 100-10-disjoint                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_ade20k100-10_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_ade20k100-10_disjoint.log)  |
| R-101-D16  | 512x512    | 100-10-overlapped                   |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_ade20k100-10_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_ade20k100-10_overlap.log)    |
| R-101-D16  | 512x512    | 100-5-disjoint                      |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_ade20k100-5_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_ade20k100-5_disjoint.log)    |
| R-101-D16  | 512x512    | 100-5-overlapped                    |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/plop/plop_r101iabnd16_aspp_512x512_ade20k100-5_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_plop) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_plop/plop_r101iabnd16_aspp_512x512_ade20k100-5_overlap.log)      |


## More

You can also download the model weights from following sources:
- BaiduNetdisk: https://pan.baidu.com/s/1e9LlD6ITuLECstFcVPaCxQ with access code **55mq**