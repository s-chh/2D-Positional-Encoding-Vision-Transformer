# ViT-2D-Positional-Embeddings
### Implementation of various 2D positional encodings for Vision Transformers.
<ul>
  <li>ViT used in a scaled-down version of the original ViT architecture from <a href="https://arxiv.org/pdf/2010.11929.pdf">An Image is Worth 16X16 Words and has only 800k parameters</a>. </li>
  <li>Works with small datasets by using a smaller patch size of 4.</li>
  <li>Implemented Positional Encodings: No Position Information, Learnable, Sinusoidal, Relative, and Rope.
</ul>  

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
    <td>No Positional Embedding</td>
    <td></td>
  </tr>
  <tr>
    <td>Learnable Positional Embedding</td>
    <td></td>
  </tr>
  <tr>
    <td>Sinusoidal Positional Embedding</td>
    <td></td>
  </tr>
  <tr>
    <td>Relative Positional Embedding</td>
    <td></td>
  </tr>
  <tr>
    <td>Rotary Position Embedding</td>
    <td></td>
  </tr>
</table>


<br><br>
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
## Training Graphs:

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
