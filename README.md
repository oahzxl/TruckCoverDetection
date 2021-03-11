# 车篷检测文档

环境初始化 服务器43005

```bash
# 安装detectron2
pip install -e Truck-Detection
# 创建数据集软连接
cd Truck-Detection
ln -s /data01/zxl/Truck-Detection/data data
```

## 检测

训练

```bash
cd ./detection/tools
python train_net.py --num-gpus 1 --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml
```

验证

```bash
cd ./detection/tools
python ./plain_train_net.py --eval-only --num-gpus 1 --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml MODEL.WEIGHTS ../outputs/faster_rcnn_R_101_FPN_3x/model.pth
```

输出卡车检测的可视化结果到../data/demo

```bash
cd ./detection/demo
rm ../../data/demo/*
python demo.py \
  --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml \
  --opts MODEL.WEIGHTS ../outputs/faster_rcnn_R_101_FPN_3x/model.pth
```

将图片中的gt截出

```bash
cd ./detection/tools
# 需要自己创建文件夹，可以在236-240行修改目标数据集和扩展大小
python ./crop_gt.py  --eval-only --num-gpus 1 --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml --eval-only
```

将图片中的预测结果截出

```bash
cd ./detection/demo
rm ../../data/demo/*
python crop.py --config-file ../configs/Truck-Detection/faster_rcnn_R_101_FPN_3x.yaml --opts MODEL.WEIGHTS ../outputs/faster_rcnn_R_101_FPN_3x/model.pth
```

## 分类

训练/测试

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode train --model res_cbam --lr 1e-4 --batch-size 16 --data-path ../data/gt_crop1 --save-path ./log/out
```

分类结果可视化

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode vis --model res_cbam --lr 1e-4 --batch-size 16 --data-path ../data/gt_crop1 --ckp-path ./log/model.tar --save-path ./log/out
```

对每张图片为分类结果，而不是每个检测框

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode vote --model res_cbam --lr 1e-4 --batch-size 16 --data-path ../data/gt_crop1 --ckp-path ./log/model.tar --save-path ./log/out
```

输出预测结果

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --mode vis --model res_cbam --lr 1e-4 --batch-size 16 --data-path ../data/gt_crop1 --ckp-path ./log/model.tar --save-path ./log/out
```