import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data/typedCSV.csv") # load data
data = data.sample(frac=1) # randomize order
labels=data['label'] # labels
data = data.iloc[:,1:] # images

training_length = int(len(labels)*0.6) # proportion for training
x_train = np.array(data.iloc[:training_length])
y_train = np.array(labels.iloc[:training_length])
x_test = np.array(data.iloc[training_length:])
y_test = np.array(labels.iloc[training_length:])

# reshape
x_train = x_train.reshape(len(y_train), 28, 28)
x_test = x_test.reshape(len(y_test), 28, 28)

# normalize
x_train=x_train / 255.0
x_test=x_test/255.0

# model hyper parameters
input_shape = (28, 28, 1)
num_classes = 62
batch_size = 64
epochs = 5

# convert labels correct ouptut type
y_train = tf.one_hot(y_train.astype(np.int32), depth=num_classes)
y_test = tf.one_hot(y_test.astype(np.int32), depth=num_classes)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])

# Early trainig stop
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])


model.save("models/charCNN.keras")

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test accuracy: {test_acc}")
print(f"test loss: {test_loss}")