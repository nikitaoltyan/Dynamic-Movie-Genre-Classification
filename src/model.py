# Nikita Oltyan
from tensorflow import keras
from tensorflow.keras.metrics import *


def make_model(input_shape, lr=1e-3):
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    metrics = [
        Accuracy(),
        Precision(),
        Recall(),
        AUC()
    ]

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        metrics=metrics,
        loss='categorical_crossentropy'
    )

    return model
