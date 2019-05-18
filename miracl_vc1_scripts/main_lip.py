import os
from scipy import misc
import numpy as np
import sys
import keras
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.utils import np_utils, generic_utils

data_dir = 'train_test_split'
X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")

X_val = np.load(f"{data_dir}/X_val.npy")
y_val = np.load(f"{data_dir}/y_val.npy")

X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

class MasterReader(object):
    def __init__(self, nc, ne, bs, lr):
        self.num_classes = nc
        self.num_epochs = ne
        self.batch_size = bs
        self.learning_rate = lr
        self.MAX_WIDTH = 90
        self.MAX_HEIGHT = 90
        self.max_seq_len = 22        


    def load_data(self):
        data_dir = 'train_test_split'
        max_seq_length = self.max_seq_len

        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []

        if os.path.exists(data_dir):
            print("Loading saved data ...")
            X_train = np.load(f"{data_dir}/X_train.npy")
            y_train = np.load(f"{data_dir}/y_train.np'Fy")

            X_val = np.load(f"{data_dir}/X_val.npy")
            y_val = np.load(f"{data_dir}/y_val.npy")

            X_test = np.load(f"{data_dir}/X_test.npy")
            y_test = np.load(f"{data_dir}/y_test.npy")

            return X_train, y_train, X_val, y_val, X_test, y_test
        else:
            people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
            data_types = ['words']
            folder_enum = ['01','02','03','04','05','06','07','08','09','10']

            UNSEEN_VALIDATION_SPLIT = ['F05']
            UNSEEN_TEST_SPLIT = ['F06']

            directory = './cropped'
            for person_id in people:
                instance_index = 0
                for data_type in data_types:
                    for word_index, word in enumerate(folder_enum):
                        print(f"Instance #{instance_index}")
                        for iteration in folder_enum:
                            path = os.path.join(directory, person_id, data_type, word, iteration)
                            filelist = sorted(os.listdir(path + '/'))
                            sequence = []
                            for img_name in filelist:
                                if img_name.startswith('color'):
                                    image = misc.imread(path + '/' + img_name)
                                    image = misc.imresize(image, (self.MAX_WIDTH, self.MAX_HEIGHT))
                                    sequence.append(image)                                         
                            pad_array = [np.zeros((self.MAX_WIDTH, self.MAX_HEIGHT))]
                            sequence.extend(pad_array * (max_seq_length - len(sequence)))
                            sequence = np.stack(sequence, axis=0)

                            if person_id in UNSEEN_TEST_SPLIT:
                                X_test.append(sequence)
                                y_test.append(instance_index)
                            elif person_id in UNSEEN_VALIDATION_SPLIT:
                                X_val.append(sequence)
                                y_val.append(instance_index)
                            else:
                                X_train.append(sequence)
                                y_train.append(instance_index)
                        instance_index += 1
                print("......")
                print('Finished reading images for person ' + person_id)

            print('Finished reading images.')

            X_train = np.array(X_train)
            X_val = np.array(X_val)
            X_test = np.array(X_test)

            y_train = np.array(y_train)
            y_val = np.array(y_val)
            y_test = np.array(y_test)
            
            print('Finished stacking the data into the right dimensions. About to start saving to disk...')

            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            np.save(data_dir+'/X_train', X_train)
            np.save(data_dir+'/y_train', y_train)
            np.save(data_dir+'/X_val', X_val)
            np.save(data_dir+'/y_val', y_val)
            np.save(data_dir+'/X_test', X_test)
            np.save(data_dir+'/y_test', y_test)
            print('Finished saving all data to disk.')

            return X_train, y_train, X_val, y_val, X_test, y_test

    def training_generator(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()

        while True:
            for i in range(int(np.shape(X_train)[0] / self.batch_size)):
                x = X_train[i * self.batch_size : (i+1) * self.batch_size]
                y = y_train[i * self.batch_size : (i+1) * self.batch_size]
                one_hot_labels = keras.utils.to_categorical(y, num_classes=self.num_classes)
                yield (x, one_hot_labels)


    def create_model(self):
        pass


if __name__ == '__main__':
    lp = MasterReader(20, 10, 50, 0.001)    
    
    X_train = np.reshape(X_train, (1300, 22, 90, 90, 1))
    X_val = np.reshape(X_val, (100, 22, 90, 90, 1))
    X_test = np.reshape(X_test, (100, 22, 90, 90, 1))
        
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    y_val = np_utils.to_categorical(y_val, 10)

    nb_filters = [8, 16]
    nb_pool = [3, 3]
    nb_conv = [7, 3]

    model = Sequential()

    model.add(Convolution3D(8, (6, 6, 6), strides = 2, input_shape=(22, 90, 90, 1), activation='relu', padding='valid'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.5))
    
#    model.add(Convolution3D(16, (3, 3, 3), strides = 2, activation='relu', padding='valid'))
#    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])

    batch_si = 50
    num_classes = 10
    num_epochs = 10
    
    from sklearn.utils import shuffle
    X, y = shuffle(X_val, y_val, random_state=0)
    

    model.fit(X_train, y_train, validation_data=(X, y), batch_size=batch_si,
          epochs=num_epochs, shuffle=True)




