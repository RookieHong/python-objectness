# Comparisons of results


## Combination of cues:

Original: 

![Original](../results/original/002053_comb_boxes.jpg)
![Original](../results/original/002053_comb_heatmap.jpg)

Python:

![Original](../results/my_impl/002053_comb_boxes.png)
![Original](../results/my_impl/002053_comb_heatmap.png)


## MS

Original: 

![Original](../results/original/002053_MS_boxes.jpg)
![Original](../results/original/002053_MS_heatmap.jpg)

Python:

![Original](../results/my_impl/002053_MS_boxes.png)
![Original](../results/my_impl/002053_MS_heatmap.png)

## CC

Original: 

![Original](../results/original/002053_CC_boxes.jpg)
![Original](../results/original/002053_CC_heatmap.jpg)

Python:

![Original](../results/my_impl/002053_CC_boxes.png)
![Original](../results/my_impl/002053_CC_heatmap.png)

## ED

Original: 

![Original](../results/original/002053_ED_boxes.jpg)
![Original](../results/original/002053_ED_heatmap.jpg)

Python:

![Original](../results/my_impl/002053_ED_boxes.png)
![Original](../results/my_impl/002053_ED_heatmap.png)

## SS

Original: 

![Original](../results/original/002053_SS_boxes.jpg)
![Original](../results/original/002053_SS_heatmap.jpg)

Python:

![Original](../results/my_impl/002053_SS_boxes.png)
![Original](../results/my_impl/002053_SS_heatmap.png)

## Speed

This speed test is run on Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz with `002053.jpg` as input image.

Original matlab implementation:

|  Cues | Time(s)|
| ---- | ---- |
| MS, CC, SS | 2.16|
| MS | 0.88 |
| CC | 1.53 |
| SS | 2.69 |

Numpy implementation:

|  Cues | Time(s)|
| ---- | ---- |
| MS, CC, SS | 5.94|
| MS | 2.10 |
| CC | 4.12 |
| SS | 8.52 |

PyTorch implementation:

|  Cues | Time(s)|
| ---- | ---- |
| MS, CC, SS | |
| MS |  |
| CC |  |
| SS | 4.40 |