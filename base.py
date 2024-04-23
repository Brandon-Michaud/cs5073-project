import os
import pickle
import wandb
import socket
import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from parser import *
from model import *


def generate_fname(args):
    '''
    Generate a file path for results based on arguments
    :param args: Command line arguments
    :return: File path
    '''
    # Indicate this is a transfer experiment
    if args.transfer:
        transfer_str = f'_transfer_{args.transfer_dataset}'
    else:
        transfer_str = ''

    # Create file path based on experiment type and datasets
    if args.exp_type == 'resnet50':
        return f'{args.results_path}/resnet50_dataset_{args.dataset}{transfer_str}'
    elif args.exp_type == 'xception':
        return f'{args.results_path}/xception_dataset_{args.dataset}{transfer_str}'
    else:
        assert False, 'Unknown experiment type'


def load_data(dataset):
    '''
    Load dataset
    :param dataset: Dataset to load
    :return: Dataset in form x_train, y_train, x_test, y_test
    '''
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        y_train = to_categorical(y_train, args.n_classes)
        y_test = to_categorical(y_test, args.n_classes)
        return x_train, y_train, x_test, y_test
    else:
        assert False, 'Unknown dataset'


def create_model(args, train_epoch_size):
    '''
    Creates a model based on the given arguments
    :param args: Command line arguments
    :param train_epoch_size: Size of a training epoch
    :return: Model
    '''
    image_size = (args.image_size[0], args.image_size[1], args.image_size[2])

    # Create learning rate
    if args.lrd:
        decay_steps = args.lrd_steps * (train_epoch_size / args.batch)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(args.lrate,
                                                                       decay_steps=decay_steps,
                                                                       decay_rate=args.lrd_rate,
                                                                       staircase=True)
    else:
        learning_rate = args.lrate

    # Create optimizer
    if args.opt == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=args.momentum, weight_decay=args.decay)
    elif args.opt == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        assert False, 'Unknown optimizer'

    # Create loss and metrics
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy()]

    # Create model
    if args.exp_type == 'resnet50':
        model = create_resnet50_model(image_size=image_size,
                                      dataset=args.dataset,
                                      transfer=args.transfer,
                                      n_classes=args.n_classes,
                                      dense_layers=args.dense,
                                      dense_activation=args.activation_dense,
                                      dropout=args.dropout,
                                      regularization=args.l2)
    elif args.exp_type == 'xception':
        model = create_xception_model(image_size=image_size,
                                      dataset=args.dataset,
                                      transfer=args.transfer,
                                      n_classes=args.n_classes,
                                      dense_layers=args.dense,
                                      dense_activation=args.activation_dense,
                                      dropout=args.dropout,
                                      regularization=args.l2)
    else:
        assert False, 'unrecognized model'

    # Compile model and return
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model


def execute_exp(args=None, multi_gpus=False):
    '''
    Execute experiment based on given arguments
    :param args: Command line arguments
    :param multi_gpus: Use multiple GPUs (boolean)
    :return: Trained model
    '''
    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch * multi_gpus

    # Load dataset
    x_train, y_train, x_test, y_test = load_data(args.transfer_dataset if args.transfer else args.dataset)

    # Create model using the command line arguments
    if multi_gpus > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = create_model(args, x_train.shape[0])
    else:
        model = create_model(args, x_train.shape[0])

    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())

    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)

    # Plot the model
    if args.render:
        render_fname = '%s_model_plot.png' % fbase
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Abort experiment if indicated by arguments
    if args.nogo:
        print("NO GO")
        return

    # Start weights and biases
    if args.wandb:
        run = wandb.init(project=args.project,
                         name=f'{args.exp_type}_dataset_{args.dataset}{f"_transfer_{args.transfer_dataset}" if args.transfer else ""}',
                         notes=fbase, config=vars(args))
        wandb.log({'hostname': socket.gethostname()})

    # Callbacks
    cbs = []
    if args.es:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.es_patience, restore_best_weights=True,
                                                          min_delta=args.es_min_delta, monitor=args.es_monitor)
        cbs.append(early_stopping_cb)
    if args.lra:
        reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor=args.lra_monitor, factor=args.lra_factor,
                                                         patience=args.lra_patience, min_delta=args.lra_min_delta)
        cbs.append(reduce_lr_cb)

    # Log training to Weights and Biases
    if args.wandb:
        wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
        cbs.append(wandb_metrics_cb)

    # Train model
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args.epochs,
                        batch_size=args.batch,
                        verbose=args.verbose >= 2,
                        validation_data=(x_test, y_test),
                        validation_steps=None,
                        callbacks=cbs)

    # Generate results data
    results = {
        'history': history
    }

    # Test set evaluation
    test_eval = model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=args.batch,
                               verbose=args.verbose >= 2)
    results['test_eval'] = test_eval

    # Log results to Weights and Biases
    if args.wandb:
        wandb.log({'final_test_loss': test_eval[0]})
        wandb.log({'final_test_sparse_categorical_accuracy': test_eval[1]})

    # Save results
    with open("%s_results.pkl" % fbase, "wb") as fp:
        pickle.dump(results, fp)

    # Save model
    if args.save_model:
        model.save("%s_model" % fbase)

    # End Weights and Biases session
    if args.wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    # Turn off GPU
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    # GPU check
    visible_devices = tf.config.get_visible_devices('GPU')
    n_visible_devices = len(visible_devices)
    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n' % n_visible_devices)
    else:
        print('NO GPU')

    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)

    # Execute experiment
    execute_exp(args, multi_gpus=n_visible_devices > 1)
