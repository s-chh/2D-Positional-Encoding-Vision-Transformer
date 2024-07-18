import os
import torch
import argparse
import datetime
from solver import Solver
from utils import print_args


def main(args):
    # Create required directories if they don't exist
    os.makedirs(args.model_path,  exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    solver = Solver(args)
    solver.train()               # Training function
    solver.plot_graphs()         # Training plots
    solver.test(train=True)      # Testing function


# Update arguments
def update_args(args):
    args.model_path  = os.path.join(args.model_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    args.n_patches   = (args.image_size // args.patch_size) ** 2
    args.is_cuda     = torch.cuda.is_available()  # Check GPU availability
    if args.is_cuda:
        print("Using GPU")
    else:
        print("Cuda not available.")

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Positional Embeddings for Vision Transformer')

    # Positional Embedding
    parser.add_argument('--pos_embed', type=str, default='learn', help='Type of Positional Embedding to Use in ViT', choices=['none', 'learn', 'sinusoidal', 'relative', 'rope'])
    parser.add_argument('--max_relative_distance', type=int, default=2, help='max relative distance used only in relative type positional embedding (referred to as k in paper)')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in the dataset')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--lr', type=float, default=5e-4, help='peak learning rate')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path to store training graphs and tsne plots')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument("--image_size", type=int, default=32, help='image size')
    parser.add_argument("--patch_size", type=int, default=4, help='patch Size')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store downloaded dataset')

    # Model Arguments
    parser.add_argument('--model_path', type=str, default='./model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=False, help="load saved model")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args = update_args(args)
    print_args(args)
    
    main(args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
