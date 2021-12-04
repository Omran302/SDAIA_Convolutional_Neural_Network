# %% codecell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D ,MaxPooling2D
from sklearn.metrics import confusion_matrix
%matplotlib inline
from cf_matrix import make_confusion_matrix
sns.set_context('talk')
# %% codecell
train_path = 'data/train/'
test_path = 'data/test/'
#%%
train_df = pd.read_csv('data/train.txt', sep=" ", header=None)
train_df.columns=['id', 'file_paths', 'labels', 'data source']
train_df=train_df.drop(['id', 'data source'], axis=1)
#%%
size = train_df.shape[0]
sample=10400/size
post_sample=400/10400
post_sample2=1000/10000

#%%
train_df = train_df.sample(frac=sample,replace=True, random_state=1,ignore_index=True)

#%%
IMG_SIZE = 250
X=[]
img=0
for file in train_df.file_paths:
    img=cv2.imread(f'{train_path}/{file}',cv2.IMREAD_GRAYSCALE)
    new_img=cv2.resize(img,(IMG_SIZE, IMG_SIZE))
    new_img=new_img/255.0
    X.append(new_img)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y = train_df['labels'].map({'negative':0, 'positive':1}).to_numpy().reshape(-1, 1)
#%%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=post_sample, random_state=255)


#%%

# %% codecell

model = Sequential()
model.add(Conv2D(64, (9, 9), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32, (6, 6)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(16, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# %% codecell
n=20
#%%
history = model.fit(X_train, y_train, epochs=n ,batch_size=32,verbose=1, validation_split=post_sample2, shuffle = True, use_multiprocessing=True)

# %% codecell
model.summary()
# %%
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
#%%
def accur():

    print('\nTest accuracy:', test_acc ,"\n")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(n)

    plt.figure(figsize=(16,9))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

#%%
accur()
#%%

#%%
