import argparse


def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='MajorProject', fromfile_prefix_chars='@')

    # High-level info for WandB
    parser.add_argument('--project', type=str, default='major_project', help='WandB project name')

    # High-level commands
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Force perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")

    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")
    parser.add_argument('--transfer', action='store_true', help='Do transfer learning')

    parser.add_argument('--label', type=str, default=None, help="Extra label to add to output files")
    parser.add_argument('--dataset', type=str, default='imagenet', help='Data set to use')
    parser.add_argument('--transfer_dataset', type=str, default='imagenet',
                        help='Data set to use for transfer learning')
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 3],
                        help="Size of input images (rows, cols, channels)")
    parser.add_argument('--n_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--opt', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum in optimizer')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay in optimizer')

    # Learning rate decay
    parser.add_argument('--lrd', action='store_true', help='Turn on learning rate decay')
    parser.add_argument('--lrd_rate', type=float, default=0.94, help='Learning rate decay rate')
    parser.add_argument('--lrd_steps', type=int, default=2,
                        help='Number of epochs after which learning rate decay is applied')

    # Learning rate annealing
    parser.add_argument('--lra', action='store_true', help='Turn on learning rate annealing')
    parser.add_argument('--lra_monitor', type=str, default='val_loss',
                        help='What to monitor for learning rate annealing')
    parser.add_argument('--lra_factor', type=float, default=0.1,
                        help='Reduction factor for learning rate annealing')
    parser.add_argument('--lra_patience', type=int, default=10, help='Patience for learning rate annealing')
    parser.add_argument('--lra_min_delta', type=float, default=0.0001,
                        help='Minimum change to not be considered plateaued in learning rate annealing')

    # Hidden unit parameters
    parser.add_argument('--dense', nargs='+', type=int, default=[],
                        help='Number of dense units per layer (sequence of ints)')
    parser.add_argument('--activation_dense', type=str, default='elu', help='Activation function for dense layers')
    parser.add_argument('--batch_normalization', action='store_true', help='Turn on batch normalization')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--l2', type=float, default=None, help="L2 regularization parameter")

    # Early stopping
    parser.add_argument('--es', action='store_true', help='Turn on early stopping')
    parser.add_argument('--es_min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--es_patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('--es_monitor', type=str, default="val_loss", help="Metric to monitor for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=10, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=3, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4,
                        help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None,
                        help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=0, help="Size of the shuffle buffer (0 = no shuffle")

    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help="Number of training batches per epoch (must use --repeat if you are using this)")

    # Post
    parser.add_argument('--render', action='store_true', default=False, help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=False, help='Save a model file')

    return parser
