python main.py --dataset cifar10 --pos_embed none
python main.py --dataset cifar10 --pos_embed learn
python main.py --dataset cifar10 --pos_embed sinusoidal
python main.py --dataset cifar10 --pos_embed relative --max_relative_distance 2
python main.py --dataset cifar10 --pos_embed rope

python main.py --dataset cifar100 --n_classes 100 --pos_embed none
python main.py --dataset cifar100 --n_classes 100 --pos_embed learn
python main.py --dataset cifar100 --n_classes 100 --pos_embed sinusoidal
python main.py --dataset cifar100 --n_classes 100 --pos_embed relative
python main.py --dataset cifar100 --n_classes 100 --pos_embed rope

python main.py --dataset fashionmnist --pos_embed none
python main.py --dataset fashionmnist --pos_embed learn
python main.py --dataset fashionmnist --pos_embed sinusoidal
python main.py --dataset fashionmnist --pos_embed relative
python main.py --dataset fashionmnist --pos_embed rope

python main.py --dataset svhn --pos_embed none
python main.py --dataset svhn --pos_embed learn
python main.py --dataset svhn --pos_embed sinusoidal
python main.py --dataset svhn --pos_embed relative
python main.py --dataset svhn --pos_embed rope


