import tensorflow as tf
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras


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
    base_model = ResNet50(weights=dataset if transfer else None,
                          include_top=False, input_shape=image_size)
    inputs = base_model.input
    outputs = base_model.output

    if transfer:
        base_model.trainable = False
    outputs = add_top(outputs, dense_layers, dense_activation, n_classes, dropout=dropout,
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
    base_model = Xception(weights=dataset if transfer else None,
                          include_top=False, input_shape=image_size)

    inputs = base_model.input
    outputs = base_model.output

    if transfer:
        base_model.trainable = False
    outputs = add_top(outputs, dense_layers, dense_activation, n_classes, dropout=dropout,
                      regularization=regularization)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
