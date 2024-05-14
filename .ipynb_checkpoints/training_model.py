import numpy as np
import cv2
# pip install opencv-python
# convert images into array
import os 
# path locations
import random
import matplotlib.pyplot as plt
import pickle
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

NAME = f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

x=x/255

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))      #64 feature detetor with size 3x3 and activation function
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))     
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))     
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))     
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = x.shape[1:], activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))  
model.add(Dense(2, activation = 'softmax'))   #return values 0 and 1

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x,y, epochs=10, validation_split=0.1, batch_size=32, callbacks=[tensorboard]) 

# no of convo layers max pool layers everthings determines the good data test
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
def predict_class(image_path=r'C:\Users\Acer\Desktop\photos\catmother.jpg'):
  """
  Preprocesses an image, makes a prediction using the trained model,
  and returns the predicted class (cat or dog) with its probability.
  """
  img = load_img(image_path, target_size=(100, 100))  # Resize the image
  img = img_to_array(img)  # Convert to array
  img = np.expand_dims(img, axis=0) # Add a new dimension for batch processing
  img = img / 255.0  # Normalize pixel values

  predictions = model.predict(img)[0]  # Make prediction and get probabilities
  class_index = predictions.argmax()  # Get the index of the highest probability class
  class_name = ('cat', 'dog')[class_index]  # Convert index to class name

  return class_name, predictions[class_index]  # Return class name and probability

# Example usage: predict class probabilities for a new image
image_path = r'C:\Users\Acer\Desktop\photos\catmother.jpg'  # Replace with your image path
class_name, probability = predict_class(r'C:\Users\Acer\Desktop\photos\catmother.jpg')

print(f'Predicted class: {class_name}, Probability: {probability:.2f}')