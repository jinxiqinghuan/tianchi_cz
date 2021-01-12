# 2

#5.切分数据集。生成训练集文件列表、验证集文件列表
import numpy as np
import os


base = 'D:/code/datasets/cz/tile_round1_train_20201231'

# 6种瑕疵的照片的标签
with open('D:/code/datasets/cz/tile_round1_train_20201231/labels.txt', 'w') as f:
    for i in range(6):
        f.write(str(i+1)+'\n')

imgs = os.listdir(base+'/train_imgs/')
np.random.seed(7)
np.random.shuffle(imgs)
val_num = int(0.1 * len(imgs))

with open(os.path.join('D:/code/datasets/cz/tile_round1_train_20201231/train.txt'), 'w') as f:
    for pt in imgs[:-val_num]:
        img = base+'train_imgs/'+pt
        ann = base+'Annotations/'+pt.replace('.jpg', '.xml')
        info = img + ' ' + ann +'\n'
        f.write(info)

with open(os.path.join('D:/code/datasets/cz/tile_round1_train_20201231/val.txt'), 'w') as f:
    for pt in imgs[-val_num:]:
        img = base+'train_imgs/'+pt
        ann = base+'Annotations/'+pt.replace('.jpg', '.xml')
        info = img + ' ' + ann + '\n'
        f.write(info)