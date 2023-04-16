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

| Backbone  | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                       |
| :-:       | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                            |
| R-101-D8  | 512x512    | 15-5-disjoint                       |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 15-5-overlapped                     |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 15-1-disjoint                       |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 15-1-overlapped                     |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 10-1-disjoint                       |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 10-1-overlapped                     |        | [cfg]() &#124; [model]() &#124; [log]()    |

