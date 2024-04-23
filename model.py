import tensorflow as tf
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomZoom, RandomRotation
from tensorflow.keras.models import Model
from tensorflow import keras


def preprocess_image(tensor):
    '''
    Preprocess image
    :param tensor: Input image
    :return: Preprocessed image tensor
    '''
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
    '''
    Add top to base model
    :param tensor: Output of base model
    :param dense_layers: Size of dense layers to add to base model; array
    :param dense_activation: Activation for dense layers
    :param n_classes: Number of output classes
    :param dropout: Probability of dropout
    :param regularization: Strength of L2 regularization
    :return: Output tensor after top has been added
    '''
    # Add regularization
    if regularization is not None:
        regularization = tf.keras.regularizers.l2(l2=regularization)

    # Get 1D tensor from convolution filters
    tensor = GlobalAveragePooling2D()(tensor)

    # Add dense layers
    for n_nodes in dense_layers:
        tensor = Dense(n_nodes, activation=dense_activation, kernel_regularizer=regularization)(tensor)
        if dropout is not None:
            tensor = Dropout(dropout)(tensor)

    # Add output layer
    tensor = Dense(n_classes, activation='softmax')(tensor)

    # Return output tensor
    return tensor


def create_resnet50_model(image_size,
                          dataset,
                          transfer,
                          n_classes,
                          dense_layers,
                          dense_activation='elu',
                          dropout=None,
                          regularization=None):
    '''
    Create resnet50 model
    :param image_size: Size of input images
    :param dataset: Dataset for pretrained weights
    :param transfer: Perform transfer learning
    :param n_classes: Number of output classes
    :param dense_layers: Size of dense layers to add to base model; array
    :param dense_activation: Activation for dense layers
    :param dropout: Probability of dropout
    :param regularization: Strength of L2 regularization
    :return: ResNet50 model
    '''
    # Input
    tensor = Input(shape=image_size)
    inputs = tensor

    # Preprocessing
    tensor = Lambda(lambda image: tf.image.resize(image, (224, 224)))(tensor)
    tensor = preprocess_image(tensor)
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

    return model


def create_xception_model(image_size,
                          dataset,
                          transfer,
                          n_classes,
                          dense_layers,
                          dense_activation='elu',
                          dropout=None,
                          regularization=None):
    '''
    Create Xception model
    :param image_size: Size of input images
    :param dataset: Dataset for pretrained weights
    :param transfer: Perform transfer learning
    :param n_classes: Number of output classes
    :param dense_layers: Size of dense layers to add to base model; array
    :param dense_activation: Activation for dense layers
    :param dropout: Probability of dropout
    :param regularization: Strength of L2 regularization
    :return: Xception model
    '''
    # Input
    tensor = Input(shape=image_size)
    inputs = tensor

    # Preprocessing
    tensor = Lambda(lambda image: tf.image.resize(image, (224, 224)))(tensor)
    tensor = preprocess_image(tensor)
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

    return model
