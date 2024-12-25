import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import ParameterGrid

# Dataset setup
def download_and_prepare_dataset():
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    local_zip = "cats_and_dogs_filtered.zip"
    tf.keras.utils.get_file(local_zip, url)

    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall("./")

    base_dir = "./cats_and_dogs_filtered"
    return os.path.join(base_dir, "train"), os.path.join(base_dir, "validation")

train_dir, validation_dir = download_and_prepare_dataset()

# Data augmentation and generators
def create_data_generators():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                             target_size=(150, 150),
                                                             batch_size=20,
                                                             class_mode='binary')

    return train_generator, validation_generator

train_generator, validation_generator = create_data_generators()

# Build the model using VGG19
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Hyperparameter tuning
def train_and_evaluate_model(learning_rate, dropout_rate, optimizer_name):
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])

    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=10,
                        validation_data=validation_generator,
                        verbose=1)
    return history.history['val_accuracy'][-1]

# Parameter grid
param_grid = {
    'learning_rate': [1e-3, 1e-4],
    'dropout_rate': [0.3, 0.5],
    'optimizer_name': ['adam', 'sgd']
}

best_accuracy = 0
best_params = None

for params in ParameterGrid(param_grid):
    accuracy = train_and_evaluate_model(**params)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_accuracy)

# Results comparison
print("Training complete. Perform comparison of results prior and after hyperparameter tuning.")
