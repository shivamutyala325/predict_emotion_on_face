from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
train_path=r'path_of_training_dataset'
test_path=r'jpath_of_testing_dataset'

#loading dataset
train_data=image_dataset_from_directory(train_path,batch_size=32,image_size=(48,48),label_mode='categorical')
test_data=image_dataset_from_directory(test_path,batch_size=32,image_size=(48,48),label_mode='categorical')

#import any pretrained model(VGG16)
import keras
from keras.layers import Flatten,Dense,Dropout
from keras.applications.vgg16 import VGG16
pre_model=VGG16(weights='imagenet',include_top=False,input_shape=(48,48,3))

for layer in pre_model.layers:
    layer.trainable=False
x=pre_model.output
x=Flatten()(x)
x=Dense(units=128,activation='relu')(x)
x=Dropout(0.1)(x)
final_layer=Dense(units=7,activation='softmax')(x)

model=keras.Model(inputs=pre_model.input,outputs=final_layer)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(train_data,epochs=30,validation_data=test_data)

#save the trained data to a file(convo_base)
model.save('conv_base.h5') 
loss,accuracy=model.evaluate(test_data)

print(f'loss={loss} ,accuracy={accuracy}')

#plotting the training data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
