from keras import models
from keras import layers
import numpy as np
from keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model(units):
    model = models.Sequential()
    model.add(layers.Dense(units, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    return model

k = 4
num_of_samples = len(train_data) // k
epochs = 100
unit_configs = [32, 64, 128]
results = {}

for units in unit_configs:
    all_scores = []

    for index in range(k):
        val_data = train_data[index * num_of_samples:(index + 1) * num_of_samples]
        val_targets = train_labels[index * num_of_samples:(index + 1) * num_of_samples]
        
        partial_train_data = np.concatenate(
            [train_data[:index * num_of_samples], train_data[(index + 1) * num_of_samples:]],
            axis=0
        )
        partial_train_labels = np.concatenate(
            [train_labels[:index * num_of_samples], train_labels[(index + 1) * num_of_samples:]],
            axis=0
        )

        model = build_model(units)
        model.fit(partial_train_data, partial_train_labels, epochs=epochs, batch_size=16, verbose=0)
        
        val_mse, _ = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mse)

    avg_score = np.mean(all_scores)
    results[units] = avg_score
    print(f'Configuration with {units} units - Validation MSE: {avg_score}')

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Average Validation MSE')
plt.title('Validation MSE for Different Hidden Unit Configurations')
plt.xticks(list(results.keys()))
plt.ylim(0, max(results.values()) * 1.1)
plt.grid(axis='y', linestyle='--')
plt.show()

best_units = min(results, key=results.get)
model = build_model(best_units)
model.fit(train_data, train_labels, epochs=epochs, batch_size=16, verbose=0)

test_mse, _ = model.evaluate(test_data, test_labels)
print(f'Test MSE with {best_units} units: {test_mse}')
