from keras import models
from keras import layers
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load IMDB dataset
max_features = 10000  # Number of words to consider as features
maxlen = 500  # Cut texts after this number of words (after padding)

# Load the IMDB dataset and keep the top `max_features` most frequent words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure uniform input size
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Function to build the model with specified number of units
def build_model(units):
    model = models.Sequential()
    model.add(layers.Dense(units, activation='tanh', input_shape=(maxlen,)))  # Hidden layer
    model.add(layers.Dense(1))  # Output layer for regression
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    return model

# Configurations for number of units
unit_configs = [32, 64]  # Different unit configurations to test
results = {}  # Dictionary to store results for each configuration
epochs = 10  # Number of training epochs

# Perform training and evaluation for each configuration
for units in unit_configs:
    # Train the model
    model = build_model(units)
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=32, verbose=0)

    # Evaluate the model on the test set
    test_mse, _ = model.evaluate(test_data, test_labels, verbose=0)
    
    # Store the average score for this configuration
    results[units] = test_mse
    print(f'Configuration with {units} units - Test MSE: {test_mse}')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Test MSE')
plt.title('Test MSE for Different Hidden Unit Configurations on IMDB Dataset')
plt.xticks(list(results.keys()))
plt.ylim(0, max(results.values()) * 1.1)  # Set y-limit for better visualization
plt.grid(axis='y', linestyle='--')
plt.show()
