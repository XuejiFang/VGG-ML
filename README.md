<div align='center' ><font size='6'>Robust Image Classification with Grayscale Sequence: A VGG-ML Fusion Model for X-Ray Pneumonia Images (ICSIP 2023, EI Index)</font></div>
<div align='center' ><font size='4'><a https://xuejifang.github.io>Xueji Fang</a>, Shanghai University</font></div>

> This is the official codebase for the paper "Robust Image Classification with Grayscale Sequence: A VGG-ML Fusion Model for X-Ray Pneumonia Images" (ICSIP 2023, EI Index) but not the whole as this VGG-ML fusion model is easy to implentation and the author is lazy to organize the codes >_<.

# Poster
![poster](./poster/poster.png)

# Implementation
## 1. 环境

PyTorch 1.8

**文件目录**

```
___ data
	|____ Chest_X-ray2-1	// 按正常、肺炎分类
	|____ Chest_X-ray2-2	// 将肺炎继续按细菌、病毒分类
	|____ Chest_X-ray3		// 按正常、细菌、病毒分类
___ models
	|____ AlexNet.py		// AlexNet模型（PyTorch）
___ Chest_X-ray.ipynb		// 测试文件
___ Split_Chest_X-ray.ipynb	// 分割细菌性和病毒性
```

> 注：X光肺炎图像数据集
>
> 链接：https://pan.baidu.com/s/1QTz8E4GtdUw3DlY0d-aNiQ 
>
> 提取码：und6 



## 2. （前期）实验结果（部分）

经典卷积神经网络效果对比

| 数据           | 模型    | epochs | val_acc  | train_loss |
| -------------- | ------- | ------ | -------- | ---------- |
| Chest_X-ray2-1 | AlexNet | 5      | 96.3964% | 0.1        |
| Chest_X-ray3   | AlexNet | 5      | 76.0590% | 0.6        |
| Chest_X-ray3   | AlexNet | 10     | 78.5317% | 0.5        |
| Chest_X-ray2-2 | AlexNet | 5      | 65.7548% | 0.6        |


# Citation
```latex
@INPROCEEDINGS{10270988,
  author={Fang, Xueji},
  booktitle={2023 8th International Conference on Signal and Image Processing (ICSIP)}, 
  title={Robust Image Classification with Grayscale Sequence: A VGG-ML Fusion Model for X-Ray Pneumonia Images}, 
  year={2023},
  volume={},
  number={},
  pages={350-354},
  doi={10.1109/ICSIP57908.2023.10270988}}
```