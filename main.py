# %% markdown
# <h1 style="text-align: center; font-family: Calibri; font-size: 36px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; font-variant: small-caps; letter-spacing: 3px; color: black; background-color: #ffffff;">COVID-19 X-Ray</h1>
# <h4 style="text-align: center; font-family: Calibri; font-size: 12px; font-style: normal; font-weight: bold; text-decoration: None; text-transform: none; letter-spacing: 1px; color: black; background-color: #ffffff;">Autor: Ali Altamimi, Faisal, Omran, Ganim - 1 / Dec / 2021</h4>
#
# %% codecell
import pandas as pd
# %% markdown
# ## Load Train Data
# %% codecell
train_df = pd.read_csv('data/train.txt', sep=" ", header=None)
train_df.columns=['id', 'file_paths', 'labels', 'data source']
train_df=train_df.drop(['id', 'data source'], axis=1 )
# %% markdown
# ## Load Train Data
# %% codecell
test_df = pd.read_csv('data/test.txt', sep=" ", header=None)
test_df.columns=['id', 'file_paths', 'labels', 'data source' ]
test_df=test_df.drop(['id', 'data source'], axis=1 )
# %% markdown
# ## EDA
# %% codecell
import matplotlib.pyplot as plt
import seaborn as sns
# %% codecell
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
# %% markdown
# ### Check the balance of the dataset
# %% codecell
#define Seaborn color palette to use
colors = sns.color_palette('pastel')
# %% codecell
# Define labels
data_lables = ['Positive', 'Negative']
# Count train and test lables
train_value_counts = train_df.labels.value_counts()
test_value_counts=test_df.labels.value_counts()
# %% codecell
fig1, ax1 = plt.subplots(1,2, figsize=(10, 5))

#create pie chart
ax1[0].pie(train_value_counts, labels = data_lables, colors = colors, autopct='%.0f%%')
ax1[0].set_xlabel('Train Dataset')

#create pie chart
ax1[1].pie(test_value_counts, labels = data_lables, colors = colors, autopct='%.0f%%')
ax1[1].set_xlabel('Test Dataset')

fig1.suptitle('Data Disterbution')
fig1.legend(data_lables)
plt.show()
# %% codecell
train_df
# %% codecell
# fig2 = sns.catplot(data = train_df, x='labels',kind='count',height=8.27, aspect=11.7/8.27, hue=)
# hue.legend()
# fig2

# %% codecell
# Spliting data based on the label
# %% codecell
# Train
train_negative_df = train_df[train_df.labels == 'negative']
train_positive_df = train_df[train_df.labels == 'positive']
# Test
test_negative_df = test_df[test_df.labels == 'negative']
test_positive_df = test_df[test_df.labels == 'positive']
# %% markdown
# ### Display Images that will be used
# %% codecell
from IPython.display import Image
import matplotlib.image as mpimg
from matplotlib import rcParams
import cv2
%matplotlib inline
# figure size in inches optional
rcParams['figure.figsize'] = 11 ,5

# %% codecell
train_path = 'data/train/'
test_path = 'data/test/'
# %% codecell
# read images
img_A = mpimg.imread(f'{train_path}/{train_negative_df.file_paths.iloc[0]}')
img_B = mpimg.imread(f'{train_path}/{train_positive_df.file_paths.iloc[4]}')
# display images
fig, ax = plt.subplots(1,2)
ax[0].imshow(img_A);
ax[0].set_xlabel('Negative Image')

ax[1].imshow(img_B);
ax[1].set_xlabel('Positive Image')
fig.suptitle('Example of Training Images')
plt.show()
# %% markdown
# From the two images we can tell that there is difference in images sizes which we need to handle later!
# %% codecell
def print_images(samples, folder_path, columns=3, rows=1, figsize=(20, 8)):
    images = samples["file_paths"].to_numpy()
    labels = samples['labels'].to_numpy()

    fig=plt.figure(figsize=figsize)

    for i, image_path in enumerate(images):
        image = cv2.imread(f'{folder_path}/{image_path}', cv2.IMREAD_COLOR)

        fig.add_subplot(rows,columns,i + 1)
        title = f'{labels[i]}'

        Sample_image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)

        plt.imshow(Sample_image, cmap='gray')
        plt.title(title)

    plt.show()
# %% codecell
print_images(train_negative_df.iloc[0:3], train_path)
print_images(train_positive_df.iloc[0:3], train_path)

# %% markdown
# Now images are resized to be in the same size
# %% codecell
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# %% codecell
from sklearn.model_selection import train_test_split

train_df_sample, valid_df_sample = train_test_split(train_df, train_size=0.9, random_state=0)
# %% codecell
print(f"Negative and positive values of train: \n{train_df_sample['labels'].value_counts()}")
print(f"Negative and positive values of validation: \n{valid_df_sample['labels'].value_counts()}")
print(f"Negative and positive values of test: \n{test_df['labels'].value_counts()}")

# %% codecell
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# %% codecell

# %% codecell
# hist = model.fit(x_train, y_train_one_hot,
#            batch_size=32, epochs=20,
#            validation_split=0.2)
# %% codecell
import numpy as np
import cv2
# %% codecell
def read_images(df):
    _list = list()
    for i in df.file_paths:
#         try:
        my_image = cv2.imread(f"{train_path}/{i}")
#         except:
#             print(i)
        _list.append(my_image)
    return np.array(_list)
# %% codecell
# read_images(train_df_sample)
# %% codecell
import glob
import cv2

images = [cv2.imread(f'{train_path}/{file}') for file in train_df_sample.file_paths]

# %% codecell
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Let's start the modelling task
# The ImageDataGenerator for keras is awesome.
#It lets you augment your images in real-time while your model is still training!
#You can apply any random transformations on each training image as it is passed to the model.
#This will not only make your model robust but will also save up on the overhead memory!


#We will apply the Image Data Generator on training with various parameters, but we won't apply
#the same parameters on testin. Why?
# Because we want the test iamges as it is, we don't want biasedness,
#also if we fit it we will be applying
# the model only on these test images only, it can't predict new images if fed into model
#Because new images will not be augmented this way


train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,
                                   shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, vertical_flip =True)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

#Now fit the them to get the images from directory (name of the images are given in dataframe) with augmentation


train_gen = train_datagen.flow_from_dataframe(dataframe = train_df_sample, directory=train_path, x_col='file_paths',
                                              y_col='labels', target_size=(200,200), batch_size=64,
                                               class_mode='binary')
valid_gen = test_datagen.flow_from_dataframe(dataframe = valid_df_sample, directory=train_path, x_col='file_paths',
                                             y_col='labels', target_size=(200,200), batch_size=64,
                                            class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(dataframe = test_df, directory=test_path, x_col='file_paths',
                                            y_col='labels', target_size=(200,200), batch_size=64,
                                             class_mode='binary')
#class mode binary because we want the classifier to predict covid or not
#target size (200,200) means we want the images to resized to 200*200
# %% codecell

# %% codecell

# %% codecell
#Now we will add some more layers to the base model for our requirements


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512)
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))


model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics=['accuracy'])

# %% codecell
history = model.fit(train_gen,batch_size=32,verbose=1, validation_split=0.2 , shuffle = True, use_multiprocessing = True)

# %% codecell

# %% codecell
