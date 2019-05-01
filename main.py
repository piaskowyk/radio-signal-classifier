# from os import path
# from pydub import AudioSegment
# import numpy as np

# files
# src = "tmp.mp3"
# dst = "test.wav"

# convert wav to mp3
# sound = AudioSegment.from_mp3(src)
# sound.export(dst, format="wav")

# sound = AudioSegment.from_file("test.wav")
# samples = sound.get_array_of_samples()
# new_sound = sound._spawn(samples)
# print(new_sound)

# song = AudioSegment.from_mp3('tmp.mp3')
# samples = song.get_array_of_samples()
# samples = np.array(samples)
#
# song2 = AudioSegment.from_wav("test.wav")
# samples2 = song2.get_array_of_samples()
# samples2 = np.array(samples2)

# for i in range(samples.size):
#     print(samples[i])

# print(samples.size)
# print(samples2.size)

#
# from keras.models import Sequential
# from keras.layers import Convolution2D, LSTM, TimeDistributed, AveragePooling1D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# import numpy as np

# classifier = Sequential()
# # classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 1), activation='relu'))
# # classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
# classifier.add(LSTM(24, input_shape=(1200, 19), return_sequences=True, implementation=2))
# classifier.add(TimeDistributed(Dense(1)))
# classifier.add(AveragePooling1D())
#
# classifier.add(Flatten())
# # classifier.add(Dense(output_dim=128, activation='relu'))
# classifier.add(Dense(1, activation='softmax'))
#
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# data = np.array([[x for x in range(1024)],
#                  [x+1 for x in range(1024)],
#                  [x+2 for x in range(1024)]])
#
# test = np.array([True, False, True])
#
# data_set = zip(data, test)
#
#
# X_train = np.reshape(data, data.shape + (1,))
#
# data_dim = 29
# timesteps = 8
# num_classes = 2
#
# model = Sequential()
# model.add(LSTM(30, return_sequences=True,
#                input_shape=data.shape[1:]))  # returns a sequence of vectors of dimension 30
# model.add(LSTM(30, return_sequences=True))  # returns a sequence of vectors of dimension 30
# model.add(LSTM(30))  # return a single vector of dimension 30
# model.add(Dense(1, activation='softmax'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model.summary()

# X_train = np.reshape(data, (data.shape[0], 1, data.shape[1]))
# test = np.reshape(test, (test.shape[0], 1, test.shape[1]))

# model.fit(data, test, batch_size = 400, epochs = 20, verbose = 1)





# data = np.expand_dims(data, axis=0)
# test = np.expand_dims(test, axis=0)

# def data_gen():
#     index = 0
#     for item in data:
#         print ("aaaaaa")
#         yield (data, {'output': test[index]})
#         index += 1

# def data_gen():
#     index = 0
#     for item in data_set:
#         print ("aaaaaa")
#         yield (item)
#         index += 1


# classifier.fit_generator(
#     data_gen(),
#     steps_per_epoch=8000,
#     epochs=2,
#     validation_steps=800
# )

# print ("mleko ==========================")
#
# result = model.predict([1, 2, 3])
# print (result)
# if result[0][0] >= 0.5:
#     print ("tak")
# else:
#     print ("nie")
import keras
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
import numpy as np

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=1024))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

# x_train = np.array([[x for x in range(1024)],
#                     [x+1 for x in range(1024)],
#                     [x + 2 for x in range(1024)],
#                     [x + 2 for x in range(1024)],
#                     [x + 2 for x in range(1024)],
#                     [x + 2 for x in range(1024)],
#                     [x + 2 for x in range(1024)],
#                     [x + 2 for x in range(1024)],
#                     [x + 2 for x in range(1024)],
#                     [x + 2 for x in range(1024)]])

x_train = np.random.rand(1000, 1024)

# y_train = keras.utils.to_categorical(np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 2]), num_classes=3)
y_train = keras.utils.to_categorical(np.random.randint(3, size=(1000, 1)), num_classes=3)

model.fit(x_train, y_train, epochs=10, batch_size=32)

# classes = model.predict_proba(np.array([x_train[0]]), batch_size=32)
# print ("predict: " + str(classes))
# classes = model.predict_classes(np.array([x_train[1]]), batch_size=32)
# print ("predict: " + str(classes))
# classes = model.predict(np.array([x_train[9]]), batch_size=32)
# print ("predict: " + str(classes))
index = 0
for item in x_train:
    classes = model.predict_classes(np.array([item]), batch_size=32)
    print (str(index) + " - predict: " + str(classes))
    index += 1