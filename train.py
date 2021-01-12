# 3

#6.训练。训练过程中可点击侧边栏的可视化，启动VisualDL，支持可视化训练过程
# 环境变量配置，用于控制是否使用GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx


train_transforms = transforms.Compose([
    transforms.RandomDistort(), transforms.RandomCrop(),
    transforms.RandomHorizontalFlip(), transforms.ResizeByShort(
        short_size=[800], max_size=1333), transforms.Normalize(
            mean=[0.5], std=[0.5]), transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.ResizeByShort(
        short_size=800, max_size=1333),
    transforms.Normalize(),
    transforms.Padding(coarsest_stride=32),
])

# 定义训练和验证所用的数据集
train_dataset = pdx.datasets.VOCDetection(
    data_dir='D:/code/datasets/cz/tile_round1_train_20201231',
    file_list='D:/code/datasets/cz/tile_round1_train_20201231/train.txt',
    label_list='D:/code/datasets/cz/tile_round1_train_20201231/labels.txt',
    transforms=train_transforms,
    num_workers=8,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='D:/code/datasets/cz/tile_round1_train_20201231',
    file_list='D:/code/datasets/cz/tile_round1_train_20201231/val.txt',
    label_list='D:/code/datasets/cz/tile_round1_train_20201231/labels.txt',
    num_workers=8,
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
# num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1
num_classes = len(train_dataset.labels) + 1

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn
model = pdx.det.FasterRCNN(
    num_classes=num_classes,
    backbone='ResNet50_vd_ssld',
    with_dcn=True,
    fpn_num_channels=64,
    with_fpn=True,
    test_pre_nms_top_n=500,
    test_post_nms_top_n=300)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
#训练8个epochs，大约需要二十个小时，可根据情况自行调整
model.train(
    num_epochs=8,
    train_dataset=train_dataset,
    train_batch_size=2,
    eval_dataset=eval_dataset,
    learning_rate=0.0025,
    lr_decay_epochs=[60, 70],
    warmup_steps=5000,
    save_dir='output/faster_rcnn_r50_vd_dcn',
    use_vdl=True)
