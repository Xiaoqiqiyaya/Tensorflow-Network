import tensorflow as tf
from tensorflow.keras import layers,datasets

def Conv1(channel,stride=2):
    return  tf.keras.Sequential([
        layers.Conv2D(channel,kernel_size=7,strides=stride,padding='same',input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(pool_size=3,strides=2,padding='same')
    ])

class Bottleneck(tf.keras.layers.Layer):
    def __init__(self,channel,stride=1,downsample=False,expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.bottleneck = tf.keras.Sequential([
            layers.Conv2D(channel,kernel_size=1,strides=1,use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channel,kernel_size=3,strides=stride,padding='same',use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channel*self.expansion,kernel_size=1,strides=1,use_bias=False),
            layers.BatchNormalization()
        ])

        if self.downsample :
            self.downsamples = tf.keras.Sequential([
                layers.Conv2D(channel*self.expansion,kernel_size=1,strides=stride,use_bias=False),
                layers.BatchNormalization()
            ])
        self.relu = layers.ReLU()

    def call(self,x):
        residual = x
        out = self.bottleneck(x)
        if self.downsample:
            residual = self.downsamples(x)
        out = out + residual
        return out

class Resnet (tf.keras.Model):
    def __init__(self,numclass = 10,expansion = 4):
        super(Resnet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(64)
        self.layer1 = self.make_layer(channel=64,block=3,stride=1)
        self.layer2 = self.make_layer(128,8,stride=2)
        self.layer3 = self.make_layer(256,36,stride=2)
        self.layer4 = self.make_layer(512,3,stride=2)
        self.avgpool = layers.AvgPool2D(7,strides=1)
        self.fc = layers.Dense(numclass)
        self.fal = layers.Flatten()


    def call(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fal(x)
        out = self.fc(x)
        return out

    def make_layer(self,channel, block, stride):
        layers = []
        layers.append(Bottleneck(channel=channel,stride=stride,downsample=True))
        for i in range(1,block):
            layers.append((Bottleneck(channel)))
        return tf.keras.Sequential(layers)
