## Introduction

<a href="https://github.com/ygjwd12345/CAF">Official Repo</a>

<a href="https://github.com/SegmentationBLWX/cssegmentation/blob/main/csseg/modules/runners/caf.py">Code Snippet</a>

<details>
<summary align="left"><a href="https://arxiv.org/pdf/2202.00432.pdf">CAF (TMM'2022)</a></summary>

```latex
@article{yang2022continual,
  title={Continual attentive fusion for incremental learning in semantic segmentation},
  author={Yang, Guanglei and Fini, Enrico and Xu, Dan and Rota, Paolo and Ding, Mingli and Hao, Tang and Alameda-Pineda, Xavier and Ricci, Elisa},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
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