# HW3: Street View House Number Recognition

## 仓库说明

dataset文件夹中，raw是原始数据，images和labels是预处理后的数据，其中原始的train数据集按照4:1的比例分割为了训练集和验证集。**（为了减小压缩包体积，删去了各文件夹内的图片）**

yolov5是YOLOv5的源代码。

preprocess.py与dataset_split.py用于数据预处理和验证集分割。这部分代码参考了[github](https://github.com/potterhsu/SVHNClassifier-PyTorch)，并做了一些改动。

get_acc.py用于统计完全正确门牌号的准确率（即预测的数字一个不多一个不少）。

**本次作业的训练结果与测试结果位于yolov5/svhn_results目录下**，包含模型checkpoint、训练过程的tensorboard记录与测试结果。



## 预处理

```bash
python preprocess.py --data-root "dataset/raw" --split "train"
python preprocess.py --data-root "dataset/raw" --split "test"
python dataset_split.py --data-root "dataset" --ratio 0.2
```



## 训练

```bash
python train.py --weights "yolov5s.pt" --cfg models/yolov5s.yaml --data data/custom-data.yaml --epochs 150 --cache --device 0 --workers 2 --project svhn_results --save-period 5 --imgsz 320 --batch-size 32
```



## 测试

```bash
python val.py --data data/custom-data.yaml --weights "svhn_results/exp6/weights/best.pt" --device 0 --project svhn_results/test
```

