import numpy as np 
import pandas as pd 
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
import os
from keras.utils.np_utils import to_categorical 

#%%
#Görsellerin manuel olarak çözünürlüklerinin düğürülmesi için değişkenler
img_rows, img_cols = 256,200

#Görsellerin bilgilerinin label değişkenine alınma aşaması
label=pd.read_csv("trainLabels.csv")

#path oluşturma
base_image_dir = os.path.join("train")

#Görsel bilgilerinin daha anlaşılır ve kullanılabilir hale getirmek için uygulanan işlemler
label['PatientId'] = label['image'].map(lambda x: x.split('_')[0])
label['eye'] = label['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0) # 1=left 0=right

#label değişkenimizin içerisinde yeni bir sütun oluşturarak görüntüleririn path'lerinin belirlenmesi.
label["path"]=label["image"].map(lambda x: os.path.join(base_image_dir, "{}.jpeg".format(x))) # paths of images

# grouphed by level and eye. After that choosed 3540 picture in 15000 imgs. All levels are same quantity.
balanced_lab = label.groupby(['level', 'eye']).apply(lambda x: x.sample(354, replace = True)).sort_index() 

#İlk durum ve görsel miktar sınırlandırması yapıldıktan sonra ki miktarların görülmesi için tablo oluşturma ve printleme işlemi
print('New Data Size:', balanced_lab.shape[0], 'Old Size:', label.shape[0]) # showing quantity of balanced data
balanced_lab[['level', 'eye']].hist(figsize = (10, 5))

#Hastalık derecesini belirleyen 0,1,2,3,4 değerlerini makine öğrenmesine verebilmek için yeni formata soktuğumuz alan
levels=balanced_lab["level"]    #catecorize 5 classes
levels=to_categorical(levels,num_classes=5)

#%%
#Path'leri belirlenen görsellerin o path üzerinden okunması
immatrix=[]
for name in balanced_lab["path"]:
    im = Image.open(name)
    img = im.resize((img_rows,img_cols))
    #gray = img.convert('L')
    immatrix.append(np.array(img).flatten())
#%% Hafızaya alnınan son görselin kullanıcıya gösterilmesi
plt.imshow(img)
#%% Normalizasyon işlemi
immatrix = img_to_array(immatrix) 
immatrix=immatrix/255 #normalization for grey form
#%% 3 boyutlu olan görsellerimizi 4 boyut'a çevirme. Sebebi sistemin öyle istemesi
immatrix=immatrix.reshape(-1,img_rows,img_cols,3) #1 for grey grey 3 for rgb
#%%
#Train test spliting
x_train,x_test,y_train,y_test=train_test_split(immatrix,levels,random_state=1,test_size=0.05)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,random_state=1,test_size=0.05)

#%% Alex-NET mimarisine ek olarak bir kaç tane daha convolution max pooling ve dense katmanlarının eklenmiş modelimiz
model = keras.models.Sequential([
    
    #filters görsel üzerinde uygulanacak filtre sayısı
    #kernal_size filtrelerin boyutları
    #strides filtrelerin kaçar satır ve sütun atlayarak filtreleme işlemini gerçekleştireceği
    #activation "relu" elde edilen verilerin 0'dan küçün ise 0'a eşitlenmesi büyük ise olduğu gibi bırakma fonksiyonu
    keras.layers.Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), activation='relu', input_shape=(img_rows,img_cols,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    keras.layers.Flatten(),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    
    keras.layers.Dense(5, activation='softmax')
])

#%%
# Forward+backward=epoch sayısı
#batch_size türkçesi parti miktarı, elde olan 3500 görüntü forward propogation işlemine kaçarlı olarak gönderileceği
epochs = 200  
batch_size = 354

# Define the optimizer
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.0, nesterov=False, name="SGD")
optimizer = keras.optimizers.Adam(lr=0.02, decay=0.005 / epochs)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# data üretme
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=40,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.3, # Randomly zoom image 5%
        shear_range=0.3,
        width_shift_range=0.3,  # randomly shift images horizontally 5%
        height_shift_range=0.3,  # randomly shift images vertically 5%
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True, 
        fill_mode="nearest")
        

datagen.fit(x_train)

# yeni verilerin üretilmediği durum
"""
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0,  # randomly rotate images in the range 5 degrees
        zoom_range = 0, # Randomly zoom image 5%
        shear_range=0,
        width_shift_range=0,  # randomly shift images horizontally 5%
        height_shift_range=0,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

"""

#%%
# Fit the model
# if stop the code before end all epochs, it will not save datas inside it.
#Öğrenme aşamasına başlanılan kısım, bu aşamaya gelene kadar hazırlanan data ve değişkenler bu aşamada modele eklenerek öğrenme başlar
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val), 
                              steps_per_epoch=((x_train.shape[0])) // batch_size)

#%%Corrolation tablosu

# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


#%% save weight
model.save_weights('learned_weights_test.h5')

#%% save history
"""import json
with open("ret_history_test","w") as f:
    json.dump(str(history.history),f)
"""
import pickle
import json
with open("ret_history_test","w") as f:
    pickle.dump(history.history, f)
    f.close()
#%% read history
import codecs
with codecs.open("ret_history_test","r",encoding="utf-8") as f:
    h=json.loads(f.read())
    
#%%
# plotting saved history file    
print(history.history.keys())    
plt.plot(h["loss"],label="train loss")
plt.plot(h["val_loss"],label="validation loss")
plt.legend()
plt.figure()
plt.plot(h["acc"],label="train acc")
plt.plot(h["val_acc"],label="validation acc")
plt.legend()
plt.show()





