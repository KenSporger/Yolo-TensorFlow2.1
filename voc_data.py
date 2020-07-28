import xml.etree.ElementTree as ET
from tensorflow.keras.utils import Sequence
import numpy as np
import math
import tensorflow as tf
import cv2


voc_root = "/home/ken/Documents/PASCAL_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
# img_sets = ['train', 'val', 'test']
img_sets = ['train', 'val']
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# xml信息读取
# return: class_id,xmin,ymin,xmax,ymax
def read_xml(path, easy=True):
    xml = open(path, 'r')
    tree = ET.parse(xml)
    root = tree.getroot()
    obj_list = []
    for obj in root.iter('object'):
        # 筛选：困难度、类别
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or (int(difficult) == 1 and easy == True):
            continue
        cls_id = classes.index(cls)
        box = obj.find('bndbox')
        b = (int(box.find('xmin').text), int(box.find('ymin').text), int(box.find('xmax').text),
             int(box.find('ymax').text))
        obj_list.append(str(cls_id)+","+ ",".join([str(a) for a in b]))
    xml.close()
    return obj_list


# 读取数据集图片信息，导出为txt
# txt: path class1,xmin1,ymin1,xmax1,ymax1 class2,xmin2,ymin2,xmax2,ymax2 ...
def read_img():
    # train val test
    for img_set in img_sets:
        # txt文件路径
        src_txt = voc_root + '/'+'ImageSets/Main/%s.txt' % img_set
        with open(src_txt, 'r') as file:
            # 去除换行符，列表存储
            img_ids = file.read().strip().split()
        # 导出数据
        dst_txt = open('%s.txt' % img_set, 'w')
        for id in img_ids:
            context = voc_root+'/JPEGImages/'+'%s.jpg' % id + ' '
            context += ' '.join(read_xml(voc_root + '/' + 'Annotations/%s.xml' % id))
            dst_txt.write(context+'\n')
        dst_txt.close()


class SequenceData(Sequence):
    def __init__(self, path, input_shape, grids, batch_size, num_classes, shuffle=True):
        """
        初始化数据发生器
        :param path: train.txt val.txt数据路径
        :param input_shape: 模型输入图片大小
        :param grids: 划分后的grid数量，如[7,7]
        :param batch_size: 一个批次大小
        :param num_classes: 类别数量
        :param shuffle: 数据乱序
        """
        # 读取txt文件
        self.datasets = []
        with open(path, "r")as f:
            self.datasets = f.readlines()
        self.input_shape = input_shape
        self.grids = grids
        self.batch_size = batch_size
        # 数据索引
        self.indexes = np.arange(len(self.datasets))
        self.is_shuffle = shuffle
        self.num_classes = num_classes

    def __len__(self):
        # 计算每一个epoch的迭代次数 ceil:3.1->4
        return math.ceil(len(self.datasets) / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # 根据索引获取datasets集合中的数据
        batch = [self.datasets[i] for i in batch_indexes]
        # 生成数据和标签
        return self.generate_data(batch)

    def get_epochs(self):
        return self.__len__()

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def generate_data(self, batch):
        # 生成批量数据
        images = []
        labels = []
        for data in batch:
            image, label  = self.split(data)
            images.append(image)
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        # output labels : (N,S,S,1+4+C)
        return [images, labels]

    def split(self, data):
        # 按空格分割
        data = data.strip().split()
        image_path = data[0]
        # 读取图片
        image = cv2.imread(image_path)
        # 获取图片原尺寸
        W = float(image.shape[1])
        H = float(image.shape[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
        # 将图片resize 到模型要求输入大小
        image = cv2.resize(image, self.input_shape)
        image = image / 255.

        labels = np.zeros(shape=(self.grids[0], self.grids[1], 5+self.num_classes))

        for box in data[1:]:
            box = box.split(",")
            temp = np.array([
                1,
                int(box[1]) / W,
                int(box[2]) / H,
                int(box[3]) / W,
                int(box[4]) / H
            ], dtype=float)
            boxes = np.concatenate((
                temp,
                np.squeeze(np.eye(1, self.num_classes, int(box[0])))
            ))
            # 填充labels
            # shape:(S,S,1+4+class)
            labels[int(boxes[1]*self.grids[0]), int(boxes[2]*self.grids[1])] = boxes

        return image, labels



if __name__ == '__main__':
    # read_img()
    a = SequenceData("./train.txt", (210, 210), (7, 7), 64, 20)
    image, labels = a.split(a.datasets[0])
    print(image.shape, labels.shape)


