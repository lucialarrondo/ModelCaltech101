#Modules
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras import utils as np_utils
from keras import models as m
from keras import layers as l

#We define some variables that will be used later
data = []
labels = []
classes = 102
cur_path = os.getcwd()
width=192
height=192

#We create a list with the names of the classes
with os.scandir(cur_path+'/101_ObjectCategories') as folders:
    names=[folder.name for folder in folders]


#Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path,'101_ObjectCategories',names[i])
    images = os.listdir(path)

    for a in images:
        try:
            img = cv2.imread(path + '\\'+ a)
            img = cv2.resize(img, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(img)
            labels.append(i)
        except:
            print("Error loading image")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


#Images of a class as an example
for i in np.where(y_train == 0)[0][:3]:
    plt.imshow(X_train[i],cmap='gray')
    plt.show()

#Data visualization
#For faster learning scale values to [0;1] boundaries
X_train_norm=X_train/255
X_test_norm=X_test/255

#We use to_categorical
y_train=np_utils.to_categorical(y_train,classes)
y_test=np_utils.to_categorical(y_test,classes)

#Building the model
model=m.Sequential([
    l.InputLayer(shape=(width,height,1)),
    l.Conv2D(filters=32,kernel_size=5,padding='valid',activation='relu'),
    l.BatchNormalization(),
    l.MaxPool2D(pool_size=2,strides=2),
    l.Dropout(rate=0.25),
    l.Conv2D(filters=64,kernel_size=3,padding='valid',activation='relu'),
    l.BatchNormalization(),
    l.MaxPool2D(pool_size=2,strides=2),
    l.Dropout(rate=0.25),
    l.Conv2D(filters=128,kernel_size=3,padding='valid',activation='relu'),
    l.BatchNormalization(),
    l.MaxPool2D(pool_size=2,strides=2),
    l.Dropout(rate=0.25),
    l.Flatten(),
    l.Dense(128, activation='relu'),
    l.BatchNormalization(),
    l.Dropout(rate=0.5),
    l.Dense(102,activation='softmax'),

])
print(model.summary())

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training the model
epochs = 15
history = model.fit(X_train_norm, y_train, batch_size=16, epochs=epochs, validation_data=(X_test_norm, y_test))
#Save the model, with an accuracy = 97% and val_accuracy=63%
model.save("model_caltech101.h5")

#Two graphs that show what happen in the training
plt.figure(0)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()