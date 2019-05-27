import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


loaded_model.load_weights("model.h5")
print("Model Loaded")
 
loaded_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("Model Compiled")
testing=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator=testing.flow_from_directory('Data/test/', 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
step_size=test_generator.n//test_generator.batch_size

score=loaded_model.evaluate_generator(generator=test_generator,steps=step_size,verbose=1)
print("Loss = ",score[0])
print("Accuracy = ",score[1])