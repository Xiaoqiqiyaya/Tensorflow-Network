import tensorflow as tf
from tensorflow.keras import layers
class AlexNet(tf.keras.Model):
    def __init__(self,num_class):
        super(AlexNet, self).__init__()
        self.feature = tf.keras.Sequential([
            layers.Conv2D(96,kernel_size=11),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=3,strides=2),
            layers.Conv2D(192,kernel_size=5,strides=1),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=3,strides=2),
            layers.Conv2D(384,kernel_size=3,strides=1),
            layers.ReLU(),
            layers.Conv2D(256,kernel_size=3,strides=1,padding="same"),
            layers.ReLU(),
            layers.Conv2D(256,kernel_size=3,strides=1,padding="same"),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=3,strides=2),
        ])

        self.classifer = tf.keras.Sequential([
            layers.Dense(4096),
            layers.Dense(4096),
            layers.Dense(num_class)
        ])
        self.flatt = tf.keras.layers.Flatten()

    def call(self,x):
        x = self.feature(x)
        x = self.flatt(x)
        x = self.classifer(x)
        return x


model = AlexNet(10)
