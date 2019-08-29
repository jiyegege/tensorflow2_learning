import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000
# 个训练样本, 10,000 个验证样本以及 25,000 个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

# TensorFlow Hub迁移模型名称
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
# 创建一个使用 Tensorflow Hub 模型嵌入（embed）语句的Keras层
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

# 创建层叠模型
model = tf.keras.Sequential()
# 添加预处理文本层
model.add(hub_layer)
# 全连接层，有16个神经元
model.add(tf.keras.layers.Dense(16, activation='relu'))
# 全连接层，有1个神经元
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

#编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#训练模型
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# 评估模型
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))