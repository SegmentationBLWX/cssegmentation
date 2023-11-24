## Introduction

<a href="https://github.com/LTTM/IL-SemSegm">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/cssegmentation/blob/main/csseg/modules/runners/ilt.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/1907.13372.pdf">ILT (ICCVW'2019)</a></summary>

```latex
@inproceedings{michieli2019incremental,
  title={Incremental learning techniques for semantic segmentation},
  author={Michieli, Umberto and Zanuttigh, Pietro},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision workshops},
  pages={0--0},
  year={2019}
}
```

</details>


## Results

#### PASCAL VOC

#### PASCAL VOC

| Backbone    | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-:         | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                                      |
| R-101-D16   | 512x512    | 15-5-disjoint                       |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-5_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_ilt) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-5_disjoint.log)    |
| R-101-D16   | 512x512    | 15-5-overlapped                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-5_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_ilt) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-5_overlap.log)      |
| R-101-D16   | 512x512    | 15-1-disjoint                       | 8.6%   | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-1_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_ilt) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-1_disjoint.log)    |
| R-101-D16   | 512x512    | 15-1-overlapped                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-1_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_ilt) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_ilt/ilt_r101iabnd16_aspp_512x512_vocaug15-1_overlap.log)      |
| R-101-D16   | 512x512    | 10-1-disjoint                       | 5.4%   | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/ilt/ilt_r101iabnd16_aspp_512x512_vocaug10-1_disjoint.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_ilt) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_ilt/ilt_r101iabnd16_aspp_512x512_vocaug10-1_disjoint.log)    |
| R-101-D16   | 512x512    | 10-1-overlapped                     |        | [cfg](https://raw.githubusercontent.com/SegmentationBLWX/cssegmentation/main/csseg/configs/ilt/ilt_r101iabnd16_aspp_512x512_vocaug10-1_overlap.py) &#124; [modellinks-per-step](https://github.com/SegmentationBLWX/modelstore/releases/tag/csseg_ilt) &#124; [log](https://github.com/SegmentationBLWX/modelstore/releases/download/csseg_ilt/ilt_r101iabnd16_aspp_512x512_vocaug10-1_overlap.log)      |

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