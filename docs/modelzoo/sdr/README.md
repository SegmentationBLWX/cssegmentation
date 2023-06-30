## Introduction

<a href="https://github.com/LTTM/SDR">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/cssegmentation/blob/main/csseg/modules/runners/sdr.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2103.06342.pdf">SDR (CVPR'2021)</a></summary>

```latex
@inproceedings{michieli2021continual,
  title={Continual semantic segmentation via repulsion-attraction of sparse and disentangled latent representations},
  author={Michieli, Umberto and Zanuttigh, Pietro},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={1114--1124},
  year={2021}
}
```

</details>


## Results

#### PASCAL VOC

| Backbone    | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                       |
| :-:         | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                            |
| R-101-D16   | 512x512    | 15-5-disjoint                       |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16   | 512x512    | 15-5-overlapped                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16   | 512x512    | 15-1-disjoint                       |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16   | 512x512    | 15-1-overlapped                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16   | 512x512    | 10-1-disjoint                       |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16   | 512x512    | 10-1-overlapped                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |

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