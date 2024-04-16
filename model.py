import tensorflow as tf
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras


def add_transfer(base_model, n_classes, dense_layers, dense_activation, dropout=None, regularization=None):
    base_model.trainable = False
    if regularization is not None:
        regularization = tf.keras.regularizers.l2(l2=regularization)

    tensor = base_model.output
    tensor = GlobalAveragePooling2D()(tensor)
    for n_nodes in dense_layers:
        tensor = Dense(n_nodes, activation=dense_activation, kernel_regularizer=regularization)(tensor)
        tensor = Dropout(dropout)(tensor)
    predictions = Dense(n_classes, activation='softmax')(tensor)
    return predictions


def create_resnet50_model(image_size, dataset, transfer, n_classes, dense_layers, dense_activation, dropout,
                          regularization, lrate, loss, metrics):
    base_model = ResNet50(weights=dataset if transfer else None,
                          include_top=not transfer, input_shape=image_size)
    inputs = base_model.input
    outputs = base_model.output

    if transfer:
        outputs = add_transfer(base_model, n_classes, dense_layers, dense_activation, dropout=dropout,
                               regularization=regularization)

    model = Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=lrate)

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


def create_xception_model(image_size, dataset, transfer, n_classes, dense_layers, dense_activation, dropout,
                          regularization, lrate, loss, metrics):
    base_model = Xception(weights=dataset if transfer else None,
                          include_top=not transfer, input_shape=image_size)

    inputs = base_model.input
    outputs = base_model.output

    if transfer:
        outputs = add_transfer(base_model, n_classes, dense_layers, dense_activation, dropout=dropout,
                               regularization=regularization)

    model = Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=lrate)

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
