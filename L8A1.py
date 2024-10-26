from keras import models
from keras import layers
from keras.datasets import reuters
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# Load Reuters dataset
num_words = 10000  # Top words to consider
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=num_words)

# Prepare the data
maxlen = 100  # Maximum length of each input
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Convert labels to categorical one-hot encoding
num_classes = np.max(train_labels) + 1  # Total number of unique classes
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Function to build the model with specified number of units
def build_model(units):
    model = models.Sequential()
    model.add(layers.Dense(units, activation='tanh', input_shape=(maxlen,)))  # Hidden layer
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for multi-class classification
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['mse'])
    return model

# Configurations for number of units
unit_configs = [32, 128]  # Different unit configurations to test
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
plt.title('Test MSE for Different Hidden Unit Configurations on Reuters Dataset')
plt.xticks(list(results.keys()))
plt.ylim(0, max(results.values()) * 1.1)  # Set y-limit for better visualization
plt.grid(axis='y', linestyle='--')
plt.show()
