import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

#mnist = tf.keras.datasets.mnist # dataset

#(x_train, y_train), (x_test, y_test) = mnist.load_data() # extrac data

#print(x_train[0])
#print(y_train[0])

data = pd.read_csv("data/typedCSV.csv") # load data
data = data.sample(frac=1) # randomize
labels=data['label']
data = data.iloc[:,1:]
training_length = int(len(labels)*0.6) # divide 60/40 split
x_train = np.array(data.iloc[:training_length])
y_train = np.array(labels.iloc[:training_length])
x_test = np.array(data.iloc[training_length:])
y_test = np.array(labels.iloc[training_length:])

x_train = x_train.reshape(len(y_train), 28, 28)
x_test = x_test.reshape(len(y_test), 28, 28)

'''
for i in range(len(y_test)):

    c = y_test[i]
    if (c>=0 and c<=9): ## numbers 0-9"
        new_c = c+48

    elif(c>=10 and c<=35): # upercase letters
        new_c = c-10+65

    elif(c>=36 and c<=61): # lowercase letters
        new_c = c-36+97

        
    img = plt.imshow(x_test[i].reshape((28,28)), cmap='grey')
    plt.title(f'The result is likely: {chr(new_c)}')
    plt.show()
'''
'''
x = []
for i in range(len(labels)):
    x.append(np.reshape(data[i], (28,28)))
'''


'''
print(np.shape(x_train[0]))
print(labels[0])

fig3 = plt.figure()
img = plt.imshow(x_train[0], cmap='grey')
print(chr(labels[0]))
plt.show()

'''

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Build a more accurate model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten image to 1D
    
    # tf.keras.layers.Dense(600, activation='relu'),  # Increase neurons
    # tf.keras.layers.BatchNormalization(),  # Normalize activations
    # tf.keras.layers.Dropout(0.1),  # Reduce overfitting
    tf.keras.layers.Dense(512, activation='relu'), #relu = rectified linear
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'), #relu = rectified linear
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),  # Output layer for classification
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(62, activation='softmax')  # Output layer for classification
])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Fit the model
model.fit(x_train, y_train, epochs=10)

accuracy, loss = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

model.save('model/chars.keras')  # Recommended Keras format

# Load trained model
model = tf.keras.models.load_model('model/chars.keras')

for i in range(len(y_train)):
    img = x_train[i] # Load as grayscale
    #img = cv.resize(img, (28, 28))  # Resize 
    #img = cv.bitwise_not(img)  # Invert colors to match MNIST format ... maybe not needed, depends on input file format (PNG 0 is white and 255 is black)
    #
    #img = cv.equalizeHist(img)  # contrast improvement
    img_in = img / 255.0  # Normalize
    prediction = model.predict(img_in.reshape(-1,784))
    c = np.argmax(prediction)

    if (c>=0 and c<=9): ## numbers 0-9"
        new_c = c+48

    elif(c>=10 and c<=35): # upercase letters
        new_c = c-10+65

    elif(c>=36 and c<=61): # lowercase letters
        new_c = c-36+97
    label = y_train[i]
    if (label>=0 and label<=9): ## numbers 0-9"
        new_label = label+48

    elif(label>=10 and label<=35): # upercase letters
        new_label = label-10+65

    elif(label>=36 and label<=61): # lowercase letters
        new_label = label-36+97


    #img = img.reshape((28, 28)) 
    plt.imshow(x_train[i].reshape((28,28)), cmap='grey')  # Remove batch and color dimensions
    plt.title(f'Prediction: {chr(new_c)}, Label: {chr(new_label)}')
    plt.show()
