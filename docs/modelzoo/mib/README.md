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

| Backbone  | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                       |
| :-:       | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                            |
| R-101-D8  | 512x512    | 15-5-disjoint                       |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 15-5-overlapped                     |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 15-1-disjoint                       |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 15-1-overlapped                     |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 10-1-disjoint                       |        | [cfg]() &#124; [model]() &#124; [log]()    |
| R-101-D8  | 512x512    | 10-1-overlapped                     |        | [cfg]() &#124; [model]() &#124; [log]()    |

