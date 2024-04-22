import tensorflow as tf
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomZoom, RandomRotation
from tensorflow.keras.models import Model
from tensorflow import keras


def preprocess_image(tensor):
    tensor = RandomFlip('horizontal')(tensor)
    tensor = RandomRotation(0.1)(tensor)
    tensor = RandomZoom(0.1)(tensor)
    return tensor


def add_top(tensor,
            dense_layers,
            dense_activation,
            n_classes,
            dropout=None,
            regularization=None):
    if regularization is not None:
        regularization = tf.keras.regularizers.l2(l2=regularization)

    tensor = GlobalAveragePooling2D()(tensor)
    for n_nodes in dense_layers:
        tensor = Dense(n_nodes, activation=dense_activation, kernel_regularizer=regularization)(tensor)
        if dropout is not None:
            tensor = Dropout(dropout)(tensor)
    tensor = Dense(n_classes, activation='softmax')(tensor)
    return tensor


def create_resnet50_model(image_size,
                          dataset,
                          transfer,
                          n_classes,
                          dense_layers,
                          dense_activation='elu',
                          dropout=None,
                          regularization=None,
                          opt=None,
                          loss=None,
                          metrics=None):
    # Input
    tensor = Input(shape=image_size)
    inputs = tensor

    # Preprocessing
    tensor = preprocess_image(tensor)
    tensor = Lambda(lambda image: tf.image.resize(image, (224, 224)))(tensor)
    tensor = tf.keras.applications.resnet50.preprocess_input(tensor)

    # Base model
    base_model = ResNet50(weights=dataset if transfer else None,
                          include_top=False, input_shape=(224, 224, 3))
    tensor = base_model(tensor)

    # Add top
    if transfer:
        base_model.trainable = False
    outputs = add_top(tensor, dense_layers, dense_activation, n_classes, dropout=dropout,
                      regularization=regularization)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


def create_xception_model(image_size,
                          dataset,
                          transfer,
                          n_classes,
                          dense_layers,
                          dense_activation='elu',
                          dropout=None,
                          regularization=None,
                          opt=None,
                          loss=None,
                          metrics=None):
    # Input
    tensor = Input(shape=image_size)
    inputs = tensor

    # Preprocessing
    tensor = preprocess_image(tensor)
    tensor = Lambda(lambda image: tf.image.resize(image, (224, 224)))(tensor)
    tensor = tf.keras.applications.xception.preprocess_input(tensor)

    # Base model
    base_model = Xception(weights=dataset if transfer else None,
                          include_top=False, input_shape=(224, 224, 3))
    tensor = base_model(tensor)

    # Add top
    if transfer:
        base_model.trainable = False
    outputs = add_top(tensor, dense_layers, dense_activation, n_classes, dropout=dropout,
                      regularization=regularization)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
