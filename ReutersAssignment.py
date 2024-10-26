from keras import models
from keras import layers
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

num_words = 10000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)

maxlen = 100
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

num_classes = np.max(train_labels) + 1
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

def build_model(units):
    model = models.Sequential()
    model.add(layers.Dense(units, activation='tanh', input_shape=(maxlen,)))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['mse'])
    return model

unit_configs = [32, 128]
results = {}
epochs = 10

for units in unit_configs:
    model = build_model(units)
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=32, verbose=0)
    test_mse, _ = model.evaluate(test_data, test_labels, verbose=0)
    results[units] = test_mse
    print(f'Configuration with {units} units - Test MSE: {test_mse}')

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Test MSE')
plt.title('Test MSE for Different Hidden Unit Configurations on Reuters Dataset')
plt.xticks(list(results.keys()))
plt.ylim(0, max(results.values()) * 1.1)
plt.grid(axis='y', linestyle='--')
plt.show()
