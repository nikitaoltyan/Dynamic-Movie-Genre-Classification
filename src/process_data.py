# Nikita Oltyan
from keras.preprocessing.image import ImageDataGenerator


def prepare_train_data(
        IMG_SHAPE=(180,180),
        BATCH_SIZE=32,
        VAL_SPLIT=0.2,
        RESCALE=1/255,
        ROTATION_RANGE=5,
        WIDTH_SHIFT_RANGE=0.1,
        HEIGHT_SHIFT_RANGE=0.1,
        ZOOM_RANGE=0.1,
        BRIGHTNESS_RANGE=(0.8, 1.2),
        HORIZONTAL_FLIP=True
    ):

    train_datagen = ImageDataGenerator(
        rescale=RESCALE,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        brightness_range=BRIGHTNESS_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        validation_split=VAL_SPLIT
    )

    # Generators
    train_generator = train_datagen.flow_from_directory(
        '../data/raw/movie_classification/train',
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        '../data/raw/movie_classification/val',
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    data_shape = (IMG_SHAPE[0], IMG_SHAPE[1], 3)

    return train_generator, val_generator, data_shape


def prepare_test_data(
        IMG_SHAPE=(180,180),
        RESCALE=1/255,
        BATCH_SIZE=1
    ):

    test_datagen = ImageDataGenerator(
        rescale=RESCALE
    )

    # Generators
    test_generator = test_datagen.flow_from_directory(
        '../data/raw/movie_classification/test',
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    data_shape = (IMG_SHAPE[0], IMG_SHAPE[1], 3)

    return test_generator, data_shape
