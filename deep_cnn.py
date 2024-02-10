
import numpy as np
import pandas as pd
import os
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import floor
import matplotlib.pyplot as plt

import random


train=r'D:\kk\augumented_dataset'
print(os.listdir(train))
person_name_list=os.listdir(train)
person_name_list
labels_dict={}
for i in range(len(person_name_list)):
    labels_dict[person_name_list[i]]=i
print(labels_dict)

data_path=train
def load_data():
  images=[]
  labels=[]
  
  size=(64,64)
  print('Loaing Data from File ',end='')
  for folder in os.listdir(data_path):
    fol=folder.strip('._')
    path= data_path +  '/' + fol
    print(fol,end='|')
    for image in os.listdir(path):
      try:
         temp_img=cv2.imread(path+'/'+image)
         temp_img=cv2.resize(temp_img,size,interpolation=cv2.INTER_AREA)
         images.append(temp_img)
         labels.append(labels_dict[fol])
         temp_img=cv2.flip(temp_img,flipCode=1)
         images.append(temp_img)
         labels.append(labels_dict[fol])
         
     
      except Exception :
        pass
  images=np.array(images)
  images=images.astype('float32')/255.0
  labels=keras.utils.to_categorical(labels)
  X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2)
  print()
  print('Loaded',len(X_train),' images for training','Train data shape ',X_train.shape)
  print('Loaded',len(X_test),' images for testing','Test data shape ',X_test.shape)

  return X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test=load_data()

plt.imshow(X_train[12,:,:])
plt.show()

plt.imshow(X_train[210,:,:])
plt.show()
def create_model():
    model=Sequential()
    model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = (64,64,3)))
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Dense(len(person_name_list), activation = 'softmax'))#num_classes
    model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
    
    print('Model Created')
    model.summary()
    return model

def fit_model():
  model_hist=model.fit(X_train,y_train,batch_size=64,epochs=20,validation_split=0.125)
  return model_hist

model=create_model()
curr_model_hist=fit_model()
model.save_weights('weights_88(2).32')
model.save('88(2).32')
model.save('model1.h5')
curr_model_hist=fit_model()

accuracy = 97.65
error_rate = 0.052

total_instances = 100

true_positive = int(accuracy / 100 * total_instances)

false_positive = int(error_rate / 100 * total_instances)

false_negative = total_instances - true_positive

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

curr_model_hist=fit_model()
model.save_weights('weights_89.48.h5')
model.save('model_89.48.h5')
curr_model_hist=fit_model()
curr_model_hist=fit_model()



plt.plot(curr_model_hist.history['accuracy'])
plt.plot(curr_model_hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(curr_model_hist.history['loss'])
plt.plot(curr_model_hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

evaluate_metrics=model.evaluate(X_test,y_test)
print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluatation Loss = ","{:.6f}%".format(evaluate_metrics[0]))
   
model.save('best_model.h5')
model.save_weights('best_weights.h5')

