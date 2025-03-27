import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#mnist = tf.keras.datasets.mnist # dataset

#(x_train, y_train), (x_test, y_test) = mnist.load_data() # extrac data

#print(x_train[0])
#print(y_train[0])

x_raw = np.loadtxt('PandP_chars.csv')
x_train = x_raw.astype('uint8')


y_train = np.loadtxt('PandP_labels.csv')
y_train = y_train.astype('int')
x_train = []
for i in range(len(y_train)):
    x_train.append(np.reshape(x_raw[i], (28,28)))

print(np.shape(x_train[0]))
print(y_train[0])

fig3 = plt.figure()
img = plt.imshow(x_train[0], cmap='grey')
print(chr(y_train[0]))
plt.show()

'''
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Build a more accurate model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten image to 1D
    tf.keras.layers.Dense(256, activation='relu'),  # Increase neurons
    tf.keras.layers.BatchNormalization(),  # Normalize activations
    tf.keras.layers.Dropout(0.2),  # Reduce overfitting
    tf.keras.layers.Dense(128, activation='relu'), #relu = rectified linear
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer for classification
])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Fit the model
model.fit(x_train, y_train, epochs=10)

accuracy, loss = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

model.save('digits.keras')  # Recommended Keras format

# Load trained model
model = tf.keras.models.load_model('digits.keras')

img = cv.imread('testimage_2.png', cv.IMREAD_GRAYSCALE)  # Load as grayscale
img = cv.resize(img, (28, 28))  # Resize 
img = cv.bitwise_not(img)  # Invert colors to match MNIST format ... maybe not needed, depends on input file format (PNG 0 is white and 255 is black)
#
img = cv.equalizeHist(img)  # contrast improvement
img = img / 255.0  # Normalize
img = img.reshape(1, 28, 28) 
prediction = model.predict(img)
print(f'The result is likely: {np.argmax(prediction)}')
plt.imshow(img.squeeze(), cmap=plt.cm.binary)  # Remove batch and color dimensions

plt.show()
'''