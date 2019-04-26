import csv
import numpy as np
import deepcut
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix

#------------------------- Read data ------------------------------
file = open('music_tu.csv', 'r',encoding = 'utf-8-sig')
data = list(csv.reader(file))
shuffle(data)

for d in data:
    print(d)

labels = [int(d[0]) for d in data]
sentences = [d[1] for d in data]

words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
for sentence in words:
    print(sentence)

#------------------- Extract bag-of-words -------------------------
vocab = set([w for s in words for w in s])
print('Vocab size = '+str(len(vocab)))

bag_of_words = np.zeros((len(words),len(vocab)))
for i in range(0,len(words)):
    count = 0
    for j in range(0,len(words[i])):
        k = 0
        for w in vocab:
            if(words[i][j] == w):
                bag_of_words[i][k] = bag_of_words[i][k]+1
                count = count+1
            k = k+1
    bag_of_words[i] = bag_of_words[i]/count

print(bag_of_words[0])

#--------------- Create feedforward neural network-----------------
inputLayer = Input(shape=(len(vocab),))
h1 = Dense(64, activation='tanh')(inputLayer)
h2 = Dense(64, activation='tanh')(h1)
outputLayer = Dense(3, activation='softmax')(h2)
model = Model(inputs=inputLayer, outputs=outputLayer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#----------------------- Train neural network-----------------------
history = model.fit(bag_of_words, to_categorical(labels), epochs=500, batch_size=50, validation_split = 0.2)


#-------------------------- Evaluation-----------------------------
y_pred = model.predict(bag_of_words[240:,:])

cm = confusion_matrix(labels[240:], y_pred.argmax(axis=1))
print('Confusion Matrix')
print(cm)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()