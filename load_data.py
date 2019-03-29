import os
from scipy import misc
import numpy as np
import sys


class MasterReader(object):
    def __init__(self, nc, ne, bs, lr):
        self.num_classes = nc
        self.num_epochs = ne
        self.batch_size = bs
        self.learning_rate = lr
        self.MAX_WIDTH = 90
        self.MAX_HEIGHT = 90

#    def training_generator(self):
#	    while True:
#		    for i in range(int(np.shape(self.X_train)[0] / self.config.batch_size)):
#                x = self.X_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
#                y = self.y_train[i * self.config.batch_size : (i + 1) * self.config.batch_size]
#		        one_hot_labels_train = keras.utils.to_categorical(y, num_classes=self.config.num_classes)
#			    yield (x,one_hot_labels_train)


#    def create_model(self, seen_validation):
#		np.random.seed(0)
#
#
#		bottleneck_train_path = 'bottleneck_features_train.npy'
#		bottleneck_val_path = 'bottleneck_features_val.npy'
#		top_model_weights = 'bottleneck_TOP_LAYER.h5'
#
#		if seen_validation is False:
#			top_model_weights = 'unseen_bottleneck_TOP_LAYER.h5'
#			#bottleneck_train_path = 'unseen_bottleneck_features_train.npy'
#			#bottleneck_val_path = 'unseen_bottleneck_features_val.npy'
#
#
#		input_layer = keras.layers.Input(shape=(self.config.max_seq_len, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
#
#		vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3))
#
#		vgg = Model(input=vgg_base.input, output=vgg_base.output)
#
#		for layer in vgg.layers[:15]:
#			layer.trainable = False
#
#		x = TimeDistributed(vgg)(input_layer)
#
#		model = Model(input=input_layer, output=x)
#
#		#bottleneck_model = Model(input=input_layer, output=x)
#
#		'''
#		if not os.path.exists(bottleneck_train_path):
#			bottleneck_features_train = bottleneck_model.predict_generator(self.training_generator(), steps=np.shape(self.X_train)[0] / self.config.batch_size)
#			np.save(bottleneck_train_path, bottleneck_features_train)
#
#		if not os.path.exists(bottleneck_val_path):
#			bottleneck_features_val = bottleneck_model.predict(self.X_val)
#			np.save(bottleneck_val_path, bottleneck_features_val)
#		'''
#
#		'''
#		conv2d1 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
#		x = TimeDistributed(conv2d1)(x) #input_shape=(self.config.max_seq_len, self.config.MAX_WIDTH, self.config.MAX_HEIGHT, 3)
#
#		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
#		x = keras.layers.core.Activation('relu')(x)
#		x = keras.layers.core.Dropout(rate=dp)(x)
#
#		pool1 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
#		x = TimeDistributed(pool1)(x)
#
#		conv2d2 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
#		x = TimeDistributed(conv2d2)(x)
#
#		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
#		x = keras.layers.core.Activation('relu')(x)
#		x = keras.layers.core.Dropout(rate=dp)(x)
#
#		pool2 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
#		x = TimeDistributed(pool2)(x)
#
#		conv2d3 = keras.layers.convolutional.Conv2D(3, 5, strides=(2,2), padding='same', activation=None)
#		x = TimeDistributed(conv2d3)(x)
#
#		x = keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(x)
#		x = keras.layers.core.Activation('relu')(x)
#		x = keras.layers.core.Dropout(rate=dp)(x)
#
#		pool3 = keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_last')
#		x = TimeDistributed(pool3)(x)
#		'''
#
#		#train_data = np.load(bottleneck_train_path)
#		#val_data = np.load(bottleneck_val_path)
#
#		input_layer_2 = keras.layers.Input(shape=model.output_shape[1:])
#
#		x = TimeDistributed(keras.layers.core.Flatten())(input_layer_2)
#
#
#		lstm = keras.layers.recurrent.LSTM(256)
#		x = keras.layers.wrappers.Bidirectional(lstm, merge_mode='concat', weights=None)(x)
#
#		#model.add(keras.layers.normalization.BatchNormalization(axis=3, momentum=0.99, epsilon=0.001))
#		#model.add(keras.layers.core.Activation('relu'))
#		x = keras.layers.core.Dropout(rate=dp)(x)
#
#		x = keras.layers.core.Dense(10)(x)
#
#		preds = keras.layers.core.Activation('softmax')(x)
#
#		model_top = Model(input=input_layer_2, output=preds)
#
#		model_top.load_weights(top_model_weights)
#
#		x = model(input_layer)
#		preds = model_top(x)
#
#		final_model = Model(input=input_layer, output=preds)
#
#
#		adam = keras.optimizers.SGD(lr=self.config.learning_rate)#, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#		final_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
#		#one_hot_labels_train = keras.utils.to_categorical(self.y_train, num_classes=self.config.num_classes)
#		one_hot_labels_val = keras.utils.to_categorical(self.y_val, num_classes=self.config.num_classes)
#
#		print('Fitting the model...')
#		'''
#		history = final_model.fit(train_data, one_hot_labels_train, epochs=self.config.num_epochs, batch_size=self.config.batch_size,\
#							validation_data=(val_data, one_hot_labels_val))
#		'''
#
#
#		history = final_model.fit_generator(self.training_generator(), steps_per_epoch=np.shape(self.X_train)[0] / self.config.batch_size,\
#					 epochs=self.config.num_epochs, validation_data=(self.X_val, one_hot_labels_val))
#
#		self.create_plots(history)
#
#
#		'''
#		print('Evaluating the model...')
#		score = model.evaluate(self.X_val, one_hot_labels_val, batch_size=self.config.batch_size)
#
#		print('Finished training, with the following val score:')
#		print(score)
#		'''
#
#
#	'''
#	def create_minibatches(self, data, shape):
#		data = [self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test]
#		for dataset in
#			batches = []
#			for i in range(0, len(data), self.config.batch_size)
#				sample = data[i:i + self.config.batch_size]
#				if len(sample) < self.config.batch_size:
#					pad = np.zeros(shape)
#					sample.extend(pad * (size - len(sample)))
#				batches.append(sample)
#	'''
#
#	def create_plots(self, history):
#	    os.mkdir('plots')
#		# summarize history for accuracy
#		plt.plot(history.history['acc'])
#		plt.plot(history.history['val_acc'])
#		plt.title('model accuracy')
#		plt.ylabel('accuracy')
#		plt.xlabel('epoch')
#		plt.legend(['train', 'validation'], loc='upper left')
#		plt.savefig('plots/acc_plot.png')
#		plt.clf()
#		# summarize history for loss
#		plt.plot(history.history['loss'])
#		plt.plot(history.history['val_loss'])
#		plt.title('model loss')
#		plt.ylabel('loss')
#		plt.xlabel('epoch')
#		plt.legend(['train', 'validation'], loc='upper left')
#		plt.savefig('plots/loss_plot.png')

    def load_data(self):
        data_dir = 'train_test_split'
        max_seq_length = 20

        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []

        if os.path.exists(data_dir):
            print("Loading saved data ...")
            X_train = np.load(f"{data_dir}/X_train.npy")
            y_train = np.load(f"{data_dir}/y_train.npy")

            X_val = np.load(f"{data_dir}/X_val.npy")
            y_val = np.load(f"{data_dir}/y_val.npy")

            X_test = np.load(f"{data_dir}/X_test.npy")
            y_test = np.load(f"{data_dir}/y_test.npy")

            return X_train, y_train, X_val, y_val, X_test, y_test
#            print('Read data arrays from disk.')
        else:

            people = ['F01','F02','F04','F05','F06','F07','F08','F09','F10','F11','M01','M02','M04','M07','M08']
            data_types = ['phrases', 'words']
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
#                            pad_array = [np.zeros((self.MAX_WIDTH, self.MAX_HEIGHT))]
#                            sequence.extend(pad_array * (max_seq_length - len(sequence)))
                            # sequence = np.array(sequence)
                            sequence = np.stack(sequence, axis=0)
                            # print(sequence.shape)
                            # print(sequence[0].shape)
                            # sys.exit()

#                            if seen_validation == False:
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
            print(np.shape(X_train))
            
            X_train = np.array(X_train[:3])
            X_val = np.array(X_val[:3])
            X_test = np.array(X_test[:3])

            print('Finished stacking the data into the right dimensions. About to start saving to disk...')

            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            np.save(data_dir+'/X_train', X_train)
            np.save(data_dir+'/y_train', np.array(y_train))
            np.save(data_dir+'/X_val', X_val)
            np.save(data_dir+'/y_val', np.array(y_val))
            np.save(data_dir+'/X_test', X_test)
            np.save(data_dir+'/y_test', np.array(y_test))
            print('Finished saving all data to disk.')

            return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    lp = MasterReader(20, 10, 50, 0.001)
    X_train, y_train, X_val, y_val, X_test, y_test = lp.load_data()
