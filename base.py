import os
import pickle
import wandb
import socket
import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow import keras

from parser import *
from model import *


def generate_fname(args):
    if args.transfer:
        transfer_str = f'_transfer_{args.transfer_dataset}'
    else:
        transfer_str = ''

    if args.exp_type == 'resnet50':
        return f'{args.results_path}/resnet50_dataset_{args.dataset}{transfer_str}'
    elif args.exp_type == 'xception':
        return f'{args.results_path}/xception_dataset_{args.dataset}{transfer_str}'
    else:
        assert False


def load_data(dataset):
    if dataset == 'cifar100':
        return keras.datasets.cifar100.load_data()


def create_model(args):
    image_size = (args.image_size[0], args.image_size[1], args.image_size[2])
    if args.exp_type == 'resnet50':
        return create_resnet50_model(image_size, args.dataset, args.transfer, args.n_classes, args.dense,
                                     args.activation_dense, args.dropout, args.l2,
                                     args.lrate, tf.keras.losses.SparseCategoricalCrossentropy(),
                                     [tf.keras.metrics.SparseCategoricalAccuracy()])
    elif args.exp_type == 'xception':
        return create_xception_model(image_size, args.dataset, args.transfer, args.n_classes, args.dense,
                                     args.activation_dense, args.dropout, args.l2,
                                     args.lrate, tf.keras.losses.SparseCategoricalCrossentropy(),
                                     [tf.keras.metrics.SparseCategoricalAccuracy()])
    else:
        assert False, 'unrecognized model'


def execute_exp(args=None, multi_gpus=False):
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch * multi_gpus

    # Create the TF datasets for training, validation, testing

    if args.verbose >= 3:
        print('Starting data flow')

    # Load individual files (all objects)
    (x_train, y_train), (x_test, y_test) = load_data(args.transfer_dataset)

    # Build the model
    if args.verbose >= 3:
        print('Building network')

    # Create the network
    if multi_gpus > 1:
        # Multiple GPUs
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            # Build network: you must provide your own implementation
            model = create_model(args)
    else:
        # Single GPU
        # Build network: you must provide your own implementation
        model = create_model(args)

    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())

    print(args)

    # Output file base and pkl file
    fbase = generate_fname(args)
    print(fbase)
    fname_out = "%s_results.pkl" % fbase

    # Plot the model
    if args.render:
        render_fname = '%s_model_plot.png' % fbase
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists" % fname_out)
        return

    #####
    # Start wandb
    run = wandb.init(project=args.project, name='%s' % args.label, notes=fbase, config=vars(args))

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Callbacks
    cbs = []
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True,
                                                      min_delta=args.min_delta, monitor=args.monitor)
    cbs.append(early_stopping_cb)

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')

    # Learn
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        verbose=args.verbose >= 2,
                        validation_data=(x_test, y_test),
                        validation_steps=None,
                        callbacks=cbs)

    # Generate results data
    # results = {}
    #
    # # Test set
    # if ds_testing is not None:
    #     print('#################')
    #     print('Testing')
    #     results['predict_testing_eval'] = model.evaluate(ds_testing)
    #     wandb.log({'final_test_loss': results['predict_testing_eval'][0]})
    #     wandb.log({'final_test_sparse_categorical_accuracy': results['predict_testing_eval'][1]})
    #
    # # Save results
    # fbase = generate_fname(args)
    # results['fname_base'] = fbase
    # with open("%s_results.pkl" % fbase, "wb") as fp:
    #     pickle.dump(results, fp)

    # Save model
    if args.save_model:
        model.save("%s_model" % fbase)

    wandb.finish()

    return model


if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose >= 3:
        print('Arguments parsed')

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

    execute_exp(args, multi_gpus=n_visible_devices)
