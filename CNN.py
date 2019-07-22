#Part 1 - BUILDING THE CNN 
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 

#Initializing Classifier 
classifier = Sequential() 

#step 1 - Convolution 
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#step 2 - Pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding 2nd layer of convolution 
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3 - Flatten 
classifier.add(Flatten())

#step 4 - Fully Connected Layer 
classifier.add(Dense(units= 128, activation = 'relu'))
classifier.add(Dense(units= 1, activation = 'sigmoid'))

#Compiling the CNN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#----------------------------------------------------------------------------------------#

#Part 2 - FITTING THE CNN TO THE IMAGES 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory('/Users/pradyutshukla/Desktop/Data Science material /Machine Learning /Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset/training_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

test_set= test_datagen.flow_from_directory( '/Users/pradyutshukla/Desktop/Data Science material /Machine Learning /Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)