import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='SNN_TL parameters')

    parser.add_argument('--name', default='SNN_TL', type=str, help='the model name')

    # data path
    parser.add_argument('--ROOT_PATH', default='./data/Office31', type=str, help='the root path of data')
    parser.add_argument('-S', '--SOURCE_NAME', default='amazon', type=str, help='the source data')
    parser.add_argument('-T', '--TARGET_NAME', default='webcam', type=str, help='the target data')

    # general
    parser.add_argument('--base_batch_size', default=20, type=int, help='the batch size of each GPU')
    parser.add_argument('--num_classes', default=31, type=int)

    # train
    parser.add_argument('--num_epochs', default=100, type=int, help='the number of epochs')
    parser.add_argument('--learning_rate', default=6e-5, type=float, help='the learning rate of train')

    # model
    parser.add_argument('--thresh', default=0.5, type=float, help='neuronal threshold')
    parser.add_argument('--lens', default=0.5, type=float, help='hyper-parameters of approximate function')
    parser.add_argument('--decay', default=0.5, type=float, help='decay constants')
    parser.add_argument('--time_window', default=12, type=int, help='the time step of SNN')

    # laplace
    parser.add_argument('-M', '--is_mix', default=1, type=int, help='mix laplace or not')
    parser.add_argument('--laplace_size', default=3, type=int, help='the number of levels in the laplace pyramid')
    parser.add_argument('--lap_threshold', default=0.3, type=float, help='the thresh of laplace spike')
    parser.add_argument('--original_ratio', default=0.1, type=float, help='the ratio of original spike')

    # mmd loss
    parser.add_argument('-F', '--mmd_function', default="linear_CKA", type=str, help='the loss function of mmd')
    parser.add_argument('-R', '--mmd_ratio', default=0.25, type=float, help='the ratio of mmd loss')
    parser.add_argument('-L', '--mmd_level', default=8, choices=[6, 7, 8], type=int,
                        help='the level of mmd loss in the structure')

    # psp
    parser.add_argument('-P', '--psp', default=0, type=int, help='use psp or not')

    return parser




