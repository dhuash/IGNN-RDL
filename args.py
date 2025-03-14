import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: gpu num or cpu.")
    parser.add_argument("--path", type=str, default="E:/论文/论文1/数据集/", help="Path of datasets.")
    parser.add_argument("--dataset", type=str, default="BlogCatalog", help="Name of datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="Fix the seed.")
    parser.add_argument("--n_repeated", type=int, default=5, help="Number of repeated times.")

    parser.add_argument("--alpha", action='store_true', default=0.8, help="alpha.")
    parser.add_argument("--beta", action='store_true', default=0.2, help="beta.")
    parser.add_argument("--lamda", action='store_true', default=1, help="lamda.")
    parser.add_argument("--gama", action='store_true', default=0.5, help="gama.")
    parser.add_argument("--fai", action='store_true', default=0.5, help="fai.")

    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--weight_decay_w1", type=float, default=5e-4, help="Weight decay w1.")
    parser.add_argument("--weight_decay_w2", type=float, default=5e-4, help="Weight decay w2.")
    parser.add_argument("--num_pc", type=int, default=20, help="Number of labeled samples per class.")
    parser.add_argument("--num_epoch", type=int, default=500, help="Number of training epochs.")
    parser.add_argument('--num_layer', type=int, default=2)

    args = parser.parse_args()

    return args
