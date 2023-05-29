import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import image_processing

### Load Datasets
train_csv = pd.read_csv("./word_dataset/written_name_train.csv")
valid_csv = pd.read_csv("./word_dataset/written_name_validation.csv")
test_csv = pd.read_csv("./word_dataset/written_name_test.csv")

### View information on the dataset to get an understanding of how it looks
print("Shape of dataset structure\n{}\n".format(train_csv.shape))
print("Overview of dataset\n{0}\n\nInformation about the dataset".format(train_csv.head()))
train_csv.info()
print()

### Find and remove null values
print(train_csv['IDENTITY'].isnull().value_counts())
print(valid_csv['IDENTITY'].isnull().value_counts())

train_csv.dropna(axis = 0, inplace = True)
valid_csv.dropna(axis = 0, inplace = True)

### Filter through all values, saving those that aren't unreadable
### Reset the index values to remove gaps after removing numerous rows
train_csv = train_csv[train_csv['IDENTITY'] != 'UNREADABLE']
valid_csv = valid_csv[valid_csv['IDENTITY'] != 'UNREADABLE']

train_csv['IDENTITY'] = train_csv['IDENTITY'].str.upper()
valid_csv['IDENTITY'] = valid_csv['IDENTITY'].str.upper()

train_csv.reset_index(inplace = True, drop = True)
valid_csv.reset_index(inplace = True, drop = True)

### Pre-processing
def load_images(path, features, size):
    img_features = []
    for i in range(size):
        img_path = path + features.loc[i, 'FILENAME']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = image_processing.roi_enhance(img)
        img = image_processing.preprocess(img)
        img_features.append(img)
    return img_features

train_size = 30000
valid_size= 3000

train_path = './word_dataset/train/train/'
valid_path = './word_dataset/validation/'

### Extract features, and labels
train_features = load_images(train_path, train_csv, train_size)
valid_features = load_images(valid_path, valid_csv, valid_size)
#train_labels = train_csv['IDENTITY']
#valid_labels = valid_csv['IDENTITY']
#print(type(train_labels))

train_features = np.array(train_features).reshape(-1, 256, 64, 1)
valid_features = np.array(valid_features).reshape(-1, 256, 64, 1)

plt.imshow(train_features[2], cmap=plt.cm.binary)
plt.show()

### Preparing labels for CTC Loss

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

name = 'JEBASTIN'
print(name, '\n',label_to_num(name))

train_labels = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps-2)
train_output = np.zeros([train_size])

for i in range(train_size):
    train_label_len[i] = len(train_csv.loc[i, 'IDENTITY'])
    train_labels[i, 0:len(train_csv.loc[i, 'IDENTITY'])]= label_to_num(train_csv.loc[i, 'IDENTITY'])    

valid_labels = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps-2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    valid_label_len[i] = len(valid_csv.loc[i, 'IDENTITY'])
    valid_labels[i, 0:len(valid_csv.loc[i, 'IDENTITY'])]= label_to_num(valid_csv.loc[i, 'IDENTITY'])    

print('True label : ',train_csv.loc[100, 'IDENTITY'] , '\ntrain_y : ',train_labels[100],'\ntrain_label_len : ',train_label_len[100], 
      '\ntrain_input_len : ', train_input_len[100])

### Build the model

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

input_data = Input(shape=(256, 64, 1), name='input')

inner = Conv2D(32, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

inner = Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)
inner = Dropout(0.3)(inner)

### CNN to RNN
inner = Reshape(target_shape=((64, 1024)), name='reshape')(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

### RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm1')(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name = 'lstm2')(inner)

### OUTPUT
inner = Dense(num_of_characters, kernel_initializer='he_normal',name='dense2')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()

### CTC Loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)

### Train the model

# the loss calculation occurs elsewhere, so we use a dummy lambda function for the loss
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate = 0.0001))

model_final.fit(x=[train_features, train_labels, train_input_len, train_label_len], y=train_output, 
                validation_data=([valid_features, valid_labels, valid_input_len, valid_label_len], valid_output),
                epochs=60, batch_size=128)

### Checking the models performance on the validation set
preds = model.predict(valid_features)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)[0][0])

prediction = []
for i in range(valid_size):
    prediction.append(num_to_label(decoded[i]))

y_true = valid_csv.loc[0:valid_size, 'IDENTITY']
correct_char = 0
total_char = 0
correct = 0

for i in range(valid_size):
    pr = prediction[i]
    tr = y_true[i]
    total_char += len(tr)
    
    for j in range(min(len(tr), len(pr))):
        if tr[j] == pr[j]:
            correct_char += 1
            
    if pr == tr :
        correct += 1 
    
print('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
print('Correct words predicted      : %.2f%%' %(correct*100/valid_size))

### Predictions on the test set

plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = './word_dataset/test/'+test_csv.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')
    
    image = image_processing.preprocess(image)
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
    
plt.subplots_adjust(wspace=0.2, hspace=-0.8)

#model_final.save('../api/ocr_ai')
model_final.save_weights('../api/model_weights.h5')