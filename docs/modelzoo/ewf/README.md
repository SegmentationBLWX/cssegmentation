## Introduction

<a href="https://github.com/schuy1er/EWF_official/">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/cssegmentation/blob/main/csseg/modules/runners/ewf.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_Endpoints_Weight_Fusion_for_Class_Incremental_Semantic_Segmentation_CVPR_2023_paper.pdf">EWF (CVPR'2023)</a></summary>

```latex
@inproceedings{xiao2023endpoints,
  title={Endpoints Weight Fusion for Class Incremental Semantic Segmentation},
  author={Xiao, Jia-Wen and Zhang, Chang-Bin and Feng, Jiekang and Liu, Xialei and van de Weijer, Joost and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7204--7213},
  year={2023}
}
```

</details>


## Results

#### PASCAL VOC

| Backbone   | Crop Size  | Setting                             | mIoU   | Download                                                                                                                                                                                                                                                                                                                                                                                       |
| :-:        | :-:        | :-:                                 | :-:    | :-:                                                                                                                                                                                                                                                                                                                                                                                            |
| R-101-D16  | 512x512    | 15-5-disjoint                       |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 15-5-overlapped                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 15-1-disjoint                       |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 15-1-overlapped                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 10-1-disjoint                       |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |
| R-101-D16  | 512x512    | 10-1-overlapped                     |        | [cfg]() &#124; [modellinks-per-step]() &#124; [log]()    |

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