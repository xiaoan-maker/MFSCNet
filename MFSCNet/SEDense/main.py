from tensorflow.keras import layers, Model, Sequential, optimizers
import matplotlib.pyplot as plt
import random, csv
import tensorflow as tf
import json
import PIL.Image as im
import numpy as np
from tensorflow import keras
from PIL import Image
import glob, os
from time import *

root = r'E:\Dataset\BreaKHis_v1'  # 数据集路径
filename = 'images.csv'  # 数据集csv文件名
batch_size = 16  # 批次大小
input_size = 224  # 输入图像大小
num_class = 8  # 分类数
epochs = 1000


# 创建图片路径和标签，并写入csv文件
def load_csv(root, filename, name2label):
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    # 如果不存在csv，则创建一个
    images = []  # 初始化存放图片路径的字符串数组
    #         for name in name2label.keys():  # 遍历所有子目录，获得所有图片的路径
    #             # glob文件名匹配模式，不用遍历整个目录判断而获得文件夹下所有同类文件
    #             # 只考虑后缀为png,jpg,jpeg的图片，比如：pokemon\\mewtwo\\00001.png
    #             images += glob.glob(os.path.join(root, name, '*.png'))
    #             images += glob.glob(os.path.join(root, name, '*.jpg'))
    #             images += glob.glob(os.path.join(root, name, '*.jpeg'))
    for name in os.listdir(os.path.join(root)):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        for path1 in os.listdir(os.path.join(root, name)):
            for path2 in os.listdir(os.path.join(root, name, path1)):
                for path3 in os.listdir(os.path.join(root, name, path1, path2)):
                    #                     if(path3==magnification):
                    images += glob.glob(os.path.join(root, name, path1, path2, path3, '*.png'))
                    images += glob.glob(os.path.join(root, name, path1, path2, path3, '*.jpg'))
                    images += glob.glob(os.path.join(root, name, path1, path2, path3, '*.jpeg'))
    #     print(len(images), images)  # 打印出images的长度和所有图片路径名
    random.shuffle(images)  # 随机打乱存放顺序
    # 创建csv文件，并且写入图片路径和标签信息
    random.shuffle(images)
    with open(os.path.join(root, filename), mode='w', newline='') as f:
        writer = csv.writer(f)
        for img in images:  # 遍历images中存放的每一个图片的路径，如pokemon\\mewtwo\\00001.png
            name = img.split(os.sep)[4]  # 用\\分隔，取倒数第二项作为类名
            label = name2label[name]  # 找到类名键对应的值，作为标签
            writer.writerow([img, label])  # 写入csv文件，以逗号隔开，如：pokemon\\mewtwo\\00001.png, 2
        print('written into csv file:', filename)
    # 读csv文件
    images, labels = [], []  # 创建两个空数组，用来存放图片路径和标签
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:  # 逐行遍历csv文件
            img, label = row  # 每行信息包括图片路径和标签
            label = int(label)  # 强制类型转换为整型
            images.append(img)  # 插入到images数组的后面
            labels.append(label)
    assert len(images) == len(labels)  # 断言，判断images和labels的长度是否相同
    return images, labels


def load_data(root, filename, mode='train'):
    # 创建数字编码表
    name2label = {}
    i = 0
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        for path1 in sorted(os.listdir(os.path.join(root, name))):
            name2label[path1] = i
            i = i + 1
    print(name2label)
    images, labels = load_csv(root, filename, name2label)  # 读取csv文件中已经写好的图片路径，和对应的标签

    if mode == 'train':  # 60%
        images = images[:int(0.8 * len(images))]
        labels = labels[:int(0.8 * len(labels))]
    elif mode == 'val':  # 20% = 60%->80%
        images = images[int(0.8 * len(images)):int(0.9 * len(images))]
        labels = labels[int(0.8 * len(labels)):int(0.9 * len(labels))]
    else:  # 20% = 80%->100%
        images = images[int(0.9 * len(images)):]
        labels = labels[int(0.9 * len(labels)):]
    return images, labels, name2label


img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])


def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean) / std
    return x


def preprocess(image_path, label):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(image_path)  # 读入图片
    x = tf.image.decode_jpeg(x, channels=3)  # 将原图解码为通道数为3的三维矩阵
    x = tf.image.resize(x, [input_size, input_size])
    # 数据增强
    x = tf.image.random_flip_up_down(x)  # 上下翻转
    # x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [input_size, input_size, 3])  # 裁剪
    x = tf.cast(x, dtype=tf.float32) / 255.  # 归一化
    x = normalize(x)
    y = tf.convert_to_tensor(label)  # 转换为张量
    #     y = tf.cast(label, dtype=tf.int32)
    return x, y


# 1.加载自定义数据集


images, labels, table = load_data(root, filename, 'train')
labels = np.array(labels)
labels = labels.reshape(-1, 1)
labels = tf.squeeze(labels, axis=1)
labels = tf.one_hot(labels, depth=num_class).numpy()
# print(labels)
db = tf.data.Dataset.from_tensor_slices((images, labels))  # images: string path， labels: number
db = db.shuffle(1000).map(preprocess).batch(batch_size)
val_images, val_labels, val_table = load_data(root, filename, 'val')
val_labels = np.array(val_labels)
val_labels = val_labels.reshape(-1, 1)
val_labels = tf.squeeze(val_labels, axis=1)
val_labels = tf.one_hot(val_labels, depth=num_class).numpy()
val_db = tf.data.Dataset.from_tensor_slices((val_images, val_labels))  # images: string path， labels: number
val_db = val_db.shuffle(1000).map(preprocess).batch(batch_size)

test_images, test_labels, test_table = load_data(root, filename, 'test')
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(-1, 1)
test_labels = tf.squeeze(test_labels, axis=1)
test_labels = tf.one_hot(test_labels, depth=num_class).numpy()
test_db = tf.data.Dataset.from_tensor_slices((test_images, test_labels))  # images: string path， labels: number
test_db = test_db.shuffle(1000).map(preprocess).batch(batch_size)


# BN -> ReLU -> 1*1 Conv -> BN -> ReLU -> 3*3 Conv
class BottleNeck(layers.Layer):
    # growth_rate对应的是论文中的增长率k，指经过一个BottleNet输出的特征图的通道数；drop_rate指失活率。
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters=4 * growth_rate,  # 使用1*1卷积核将通道数降维到4*k
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=growth_rate,  # 使用3*3卷积核，使得输出维度（通道数）为k
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")
        self.dropout = layers.Dropout(rate=drop_rate)

        #         SE-block
        self.se_gap = layers.GlobalAveragePooling2D()
        self.se_avgp = layers.AveragePooling2D(pool_size=3, strides=3)
        #         self.se_maxp = layers.MaxPooling2D(pool_size=3,strides=3)
        self.se_resize = layers.Reshape((1, 1, growth_rate))
        self.se_fc1 = layers.Dense(units=growth_rate // 16, activation=tf.keras.activations.relu)
        self.se_fc2 = layers.Dense(units=growth_rate, activation=tf.keras.activations.sigmoid)

        # 将网络层存入一个列表中
        self.listLayers = [self.bn1,
                           layers.Activation("relu"),
                           self.conv1,
                           self.bn2,
                           layers.Activation("relu"),
                           self.conv2,
                           self.dropout]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)

        #         SE-block
        b = self.se_gap(y)
        b = self.se_resize(b)

        c = y
        c = self.se_avgp(c)
        #         print("avgp:",c)
        c = layers.Reshape((1, 1, c.shape[1] * c.shape[2] * c.shape[3]))(c)
        #         print("avgpre:",c)

        #         #方差
        #         e = y
        #         e,f = tf.nn.moments(e,axes=[1,2])
        # #         print("方差1:",e)
        # #         print("方差2:",f)
        #         f = layers.Reshape((1,1,b.shape[-1]))(f)
        # #         print("tr:",e)

        #         d = tf.concat([b,c,f],axis=-1)
        d = tf.concat([b, c], axis=-1)

        #         print("re:",b)
        d = self.se_fc1(d)
        #         print("fc1:",b)
        d = self.se_fc2(d)
        #         print("fc2:",b)
        y = layers.Multiply()([y, d])

        # 每经过一个BottleNet，将输入和输出按通道连结。作用是：将前l层的输入连结起来，作为下一个BottleNet的输入。
        y = layers.concatenate([x, y], axis=-1)  # 第一次3+...
        return y


# 稠密块，由若干个相同的瓶颈层构成
# BottleNeck * 6
class DenseBlock(layers.Layer):
    # num_layers表示该稠密块存在BottleNet的个数，也就是一个稠密块的层数L
    # 121为：6， 12， 24， 16
    def __init__(self, num_layers, growth_rate, drop_rate=0.5):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.listLayers = []
        # 一个DenseBlock由多个相同的BottleNeck构成，我们将它们放入一个列表中。
        for _ in range(num_layers):
            self.listLayers.append(BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionLayer(layers.Layer):
    # out_channels代表输出通道数
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = layers.BatchNormalization()  # BN
        self.conv = layers.Conv2D(filters=out_channels,  # 1*1 ConV
                                  kernel_size=(1, 1),
                                  strides=1,
                                  padding="same")
        self.pool = layers.AveragePooling2D(pool_size=(2, 2),  # 2倍下采样
                                            strides=2,
                                            padding="same")
        self.se_gap = layers.GlobalAveragePooling2D()
        self.se_avgp = layers.AveragePooling2D(pool_size=3, strides=3)
        #         self.se_maxp = layers.MaxPooling2D(pool_size=3,strides=3)
        self.se_resize = layers.Reshape((1, 1, out_channels))
        self.se_fc1 = layers.Dense(units=out_channels // 16, activation=tf.keras.activations.relu)
        self.se_fc2 = layers.Dense(units=out_channels, activation=tf.keras.activations.sigmoid)

    def call(self, inputs):
        x = self.bn(inputs)
        x = tf.keras.activations.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        b = self.se_gap(x)
        b = self.se_resize(b)
        c = x
        c = self.se_avgp(c)
        c = layers.Reshape((1, 1, c.shape[1] * c.shape[2] * c.shape[3]))(c)
        d = tf.concat([b, c], axis=-1)
        d = self.se_fc1(d)
        d = self.se_fc2(d)
        d = layers.Multiply()([x, d])
        return d


# DenseNet整体网络结构
class DenseNet(tf.keras.Model):
    # num_init_features:代表初始的通道数，即输入稠密块时的通道数
    # growth_rate:对应的是论文中的增长率k，指经过一个BottleNet输出的特征图的通道数
    # block_layers:每个稠密块中的BottleNet的个数
    # compression_rate:压缩因子，其值在(0,1]范围内
    # drop_rate：失活率
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super(DenseNet, self).__init__()
        # 第一层，7*7的卷积层，2倍下采样。
        self.conv = layers.Conv2D(filters=num_init_features,
                                  kernel_size=(7, 7),
                                  strides=2,
                                  padding="same")
        self.bn = layers.BatchNormalization()
        # 最大池化层，3*3卷积核，2倍下采样
        self.pool = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        # 稠密块 Dense Block(1)
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        # 该稠密块总的输出的通道数
        self.num_channels += growth_rate * block_layers[0]
        # 对特征图的通道数进行压缩
        self.num_channels = compression_rate * self.num_channels
        # 过渡层1，过渡层进行下采样
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))
        # 稠密块 Dense Block(2)
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        # 过渡层2，2倍下采样，输出：14*14
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(3)
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        # 过渡层3，2倍下采样
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(4)
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)

        # 全局平均池化，输出size：1*1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 全连接层，进行10分类
        # 分类个数 units = ？
        self.fc = layers.Dense(units=num_class, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        x = self.conv(inputs)
        #         print(x)
        x = self.bn(x)
        x = tf.keras.activations.relu(x)
        x = self.pool(x)
        x = self.dense_block_1(x)
        x = self.transition_1(x)
        x = self.dense_block_2(x)
        x = self.transition_2(x)
        x = self.dense_block_3(x)
        x = self.transition_3(x, )
        x = self.dense_block_4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x


model = DenseNet(num_init_features=input_size, growth_rate=16, block_layers=[6, 12, 24, 16], compression_rate=0.5,
                 drop_rate=0.5)

model.build(input_shape=(None, input_size, input_size, 3))
model.summary()

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=METRICS)

history = model.fit(
    db,
    epochs=epochs,
    validation_data=val_db
)
