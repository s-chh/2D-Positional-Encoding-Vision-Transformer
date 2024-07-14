# Positional Embeddings for Vision Transformers (ViT)
Implemented Positional Encodings: <strong>No Position</strong>, <strong>Learnable</strong>, <strong>Sinusoidal (Absolute)</strong>, <strong>Relative</strong>, and <strong>Rotary (RoPe)</strong>.
<ul>
  <li>Works by splitting dimensions into two parts and implements 1D positional embeddings on each part.</li>
  <li>One part uses the x-positions sequence, and the other uses y-positions.</li>
  <li>Classification token is handled differently in all methods. Check below for more details.</li>
  <li>Network used here in a scaled-down version of the original ViT with only 800k parameters</a>. </li>
  <li>Works with small datasets by using a smaller patch size of 4.</li>
</ul>  
<br>

## Run commands (also available in <a href="scripts.sh">scripts.sh</a>): <br>
Different positional embeddings can be chosen using the <strong>pos_embed</strong> argument. Example:
<table>
  <tr>
    <th>Positional Embedding Type</th>
    <th>Run command</th>
  </tr>
  <tr>
    <td>No Position</td>
    <td>python main.py --dataset cifar10 --pos_embed <strong>none</strong></td>
  </tr>
  <tr>
    <td>Learnable</td>
    <td>python main.py --dataset cifar10 --pos_embed <strong>learn</strong></td>
  </tr>
  <tr>
    <td>Sinusoidal</td>
    <td>python main.py --dataset cifar10 --pos_embed <strong>sinusoidal</strong></td>
  </tr>
  <tr>
    <td>Relative</td>
    <td>python main.py --dataset cifar10 --pos_embed <strong>relative</strong></td>
  </tr>
  <tr>
    <td>Rotary (Rope) </td>
    <td>python main.py --dataset cifar10 --pos_embed <strong>rope</strong></td>
  </tr>
</table>
Change dataset to appropriate dataset.
<br>

## Results
<table>
  <tr>
    <th>Positional Encoding Type</th>
    <th>FashionMNIST</th>
    <th>SVHN</th>
    <th>CIFAR10</th>
    <th>CIFAR100</th>
  </tr>
  <tr>
    <td>No Position</td>
    <td></td>
  </tr>
  <tr>
    <td>Learnable</td>
    <td></td>
  </tr>
  <tr>
    <td>Sinusoidal</td>
    <td></td>
  </tr>
  <tr>
    <td>Relative</td>
    <td></td>
  </tr>
  <tr>
    <td>Rotary Position Embedding (Rope) </td>
    <td></td>
  </tr>
</table>
<br>

## Splitting 2D to Mutiple 1D Positonal Embeddings:

## Handling Classification Token:
<ul>
<li>No Position: Nos</li>
<li>Learnable: Learns classification Token, which includes its positional embedding.</li>
<li>Sinusoidal: No positional embedding is added to the classification token.</li>
<li>Relative: One solution is not to update the classification token. Instead, a separate index (used 0 here) in the embedding lookup tables represents distances to the Classification token. </li>  
<li>Rotatory: Sequences of patches start at 1 (instead of 0), and 0 represents the position of the classification token. Using a 0 index for classification token results in no change/rotation.</li>
</ul>

## Transformer Config:
<table>
  <tr>
    <td>Input Size</td>
    <td> 3 X 32 X 32  </td>
  </tr>
  <tr>
    <td>Patch Size</td>
    <td>4</td>
  </tr>
  <tr>
    <td>Sequence Length</td>
    <td>8*8 = 64</td>
  </tr>
  <tr>
    <td>Embedding Size </td>
    <td>128</td>
  </tr>
  <tr>
    <td>Num of Layers </td>
    <td>6</td>
  </tr>
  <tr>
    <td>Num of Heads </td>
    <td>4</td>
  </tr>
  <tr>
    <td>Forward Multiplier </td>
    <td>2</td>
  </tr>
  <tr>
    <td>Dropout </td>
    <td>0.1</td>
  </tr>
  <tr>
    <td>Parameters </td>
    <td>820k</td>
  </tr>
</table>

<!--
<br><br>
### Training Graphs:

<table>
  <tr>
    <th>Dataset</th>
    <th>Accuracy</th>
    <th>Loss</th>
  </tr>
  <tr>
    <td>MNIST</td>
    <td> <img src="outputs/mnist/graph_accuracy.png"  alt="MNIST_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/mnist/graph_loss.png"  alt="MNIST_loss" width = 500px height = 250px ></td>
  </tr>
  <tr>
    <td>FMNIST</td>
    <td> <img src="outputs/fmnist/graph_accuracy.png"  alt="FMNIST_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/fmnist/graph_loss.png"  alt="FMNIST_loss" width = 500px height = 250px ></td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td> <img src="outputs/svhn/graph_accuracy.png"  alt="SVHN_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/svhn/graph_loss.png"  alt="SVHN_loss" width = 500px height = 250px ></td>
  </tr>
  <tr>
    <td>CIFAR10</td>
    <td> <img src="outputs/cifar10/graph_accuracy.png"  alt="CIFAR10_accuracy" width = 500px height = 250px> </td>
    <td> <img src="outputs/cifar10/graph_loss.png"  alt="CIFAR10_loss" width = 500px height = 250px ></td>
  </tr>
</table>
-->
