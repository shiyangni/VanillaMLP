## Project Context

In `VanillaMLP` we created a framework that allows users to create and train a basic Multi-Layer Perceptron model in C++. 

This is a practice project to gain an elementary understanding of how a Neural Network is implemented. Our implementation is mostly naïve, and very likely has serious flaws. This mini report aims to clearly describe the key components in our design. By doing so we hope to expose our current level of understanding (or the lack thereof), and invite _directed_ critical feedback. Your kindness in helping us further  learning would be highly appreciated!

## How to run `VanillaMLP`

### Prerequisites

- We used the non-standard `#pragrama once` as include guard. Most popular compilers support this directive. We recommend using any GCC compiler with versions later than 3.4.

- Matrix algebra is implemented through library `Eigen 3`. The easiest way to install `Eigen 3` is through Microsoft's C++ package manager `vcpkg`. Please refer to https://github.com/Microsoft/vcpkg for `vcpkg` installation. 

- After installing `vcpkg`, `cd` to its hosting directory, and run one of the following depending on which shell you're using:

  ```
  PS> .\vcpkg install eigen3
  Linux:~/$ ./vcpkg install sdl2 curl
  ```

### Configuring Source Files

Clone the source file repo through

```
git clone https://github.com/shiyangni/VanillaMLP.git
```

Once complete, open an C++ IDE of your choice , start an empty project, and add all `.h` and `.cpp` files according to your project structure. In Visual Studio C++, that simply involves navigating to the `Solution Explorer` bar (typically appearing on the right of the IDE), right clicking on `Header` and `Source` folder, and adding all relevant files through  `Add Existing Item`.  

Note no changes to the file structure is necessary. we also included a `DataGeneration.ipynb` that contains code for producing simulated data. To ensure the produced data is accessible through `main.cpp` we recommend preserving the file structure _as is_.

### Reading In Data

In the `main` function in `Main.cpp`, use the following command to read in data.

```c++
#include <Eigen/Dense>
#include "Utilities.h"  // contains readCSV.
Eigen::MatrixXd data = readCSV("data.csv")
```

Our current iteration imposes the following restrictions on the inputting data:

- The data can only contain numerical values coercible into `double`.
- The first row of  `data.csv` has to be a header.
- The first column of `data.csv` contains the labels, and the rest features. Simulated data generated from `DataGeneration.ipynb` is automatically in that format.

### Initiating A Model

Initiate a model with the following code:

```c++
#include <Eigen/Dense>
#include "Model.h"
#include "Utilities.h" // contains self-defined activation function.

using namespace std;
using namespace Eigen;

Model m(data); // Initiate a model and configure the dimensions of the input layer.
m.addHiddenLayer(numOfOutputs=, activation=); // Activation has to be a std::function<double(double)>. We supply several activations in Utilities.h, including sigmoid, identity and bentIdentity. Please refer to the header file for documentation.
m.addHiddenLayer(numOfOutputs=, activation=);
...; // Add custom number of layers.
m.addOutputLayer(); // Add the output layer to signal completion of model construction.
 
```

Our current iteration has the following constraints on models. Some of these constraints can be relaxed through minor modifications.

- __No categorical input__ is supported. User needs to manually one-hot encode factor inputs.

- Our model can currently _only_ handle a __scalar valued regression problem__. This is because customized `OutputLayer` isn't supported. Currently, an `OutputLayer` can only produce output by summing over the inputs from the last `HiddenLayer`. 

  To extend the model's applicability to classification problems, we can extend the `calcOutput()` method in `OutputLayer` class to include a choice for probability coercion mapping, whereby the output vector from the last `HiddenLayer` is passed through a _SoftMax_ function.

- No weights and biases can be left out in parameter updating. This means __every layer is fully connected__.

  The fix is simple. We can add two _Change Indicator_ matrices to each layer -- one for weights and one for biases.  They have the same dimensions as the weights and biases matrices, with 1 indicating the element at that position should be included in parameter updating. Upon updating the parameters, element-wise product the gradients with the _Change Indicator_ matrices to obtain the new updating factor.

- __No regularization__ (such as drop-out or weight-decay) __or normalization__ is implemented.

- __No custom parameter initialization__ is supported. Currently all parameters are initialized to 1 for convenience in implementation.  But this proved disastrous on data with feature magnitudes much larger than 1 -- our test shows any non-linear structure would likely get stuck at a local minimum. 

  To avoid this situation, we recommend _normalizing the dataset_ before passing in.

### Training and Prediction

Training a model is as simple as

```c++
m.train_sgd(data, numOfEpochs, learningRate, miniBatchSize); 
// data is an read-in from readCSV(). Again the first column has to be labels, and the rest features.
// In each epoch, the floorDiv(dataSize, miniBatchSize) number of mini batches are selected, and the rest of the dataset is truncated.
```

In each epoch, the current training MSE and time elapsed would be reported. A part of the sample output looks like this

```
...
Epoch: 263, Loss: 17.0335, Time Elapsed: 20 milliseconds
Epoch: 264, Loss: 16.299, Time Elapsed: 19 milliseconds
Epoch: 265, Loss: 15.6037, Time Elapsed: 19 milliseconds
Epoch: 266, Loss: 14.9456, Time Elapsed: 21 milliseconds
...
```

To predict on test data, simply run

```c++
m.predict(testData)
```

The `testData` has to be read-in through `readCSV` and formatted accordingly.

We didn't implement `m.score(testData)`.

## Testing the Framework

### Correctness Test

We first test the correctness of the gradient optimization scheme.

Note we __CANNOT__ equate the _correctness of the gradient descent_ with the convergence of _any model_ on any dataset. Some models might get stuck at a local minimum because of the multi-modalness of the loss function implied by the model structure and input data.

To eliminate such confounding factor, we need __a particular model structure__, whose loss on __a particular dataset__ will converge _as long as the learning rate is appropriate_. 

__A linear structure on a linearly generated data with no error__ fits the criteria. When activation functions in all layers are set to the identity, an MLP reduces to a linear regression model. The classical OLS procedure projects `Y` on the column space of `X`. When `Y` is exactly a linear combination of columns of`X`,  the projection should be exact, meaning the loss is zero. Thus, running any linearly structured MLP on a linearly generated data with no error, the loss should converge to 0 under the appropriate learning rate.

We three linear structures.

- The first one directly models a regression: it has one `HiddenLayer` with one neuron, activated by identity.
- The second one has one `HiddenLayer` but two neurons, activated by identity.
- The third one has two `HiddenLayers`, each with two neurons, both activated by identity.

The loss on the first model converges to 0:

```
The first model has the simplest linear regression structure, i.e, one hiddenLayer with one neuron, activated through identity.
...
Epoch: 0, Loss: 5.1609e+11, Time Elapsed: 47 milliseconds
Epoch: 1, Loss: 2.12799e+11, Time Elapsed: 34 milliseconds
Epoch: 2, Loss: 9.2833e+10, Time Elapsed: 30 milliseconds
Epoch: 3, Loss: 4.32552e+10, Time Elapsed: 26 milliseconds
Epoch: 4, Loss: 2.16893e+10, Time Elapsed: 28 milliseconds
Epoch: 5, Loss: 1.17526e+10, Time Elapsed: 22 milliseconds
...
Epoch: 495, Loss: 1.6952, Time Elapsed: 19 milliseconds
Epoch: 496, Loss: 1.69453, Time Elapsed: 19 milliseconds
Epoch: 497, Loss: 1.69386, Time Elapsed: 19 milliseconds
Epoch: 498, Loss: 1.6932, Time Elapsed: 19 milliseconds
Epoch: 499, Loss: 1.69254, Time Elapsed: 19 milliseconds
```

On the second model, the training error converges to zero with a slower rate:

```
The second linear model has 1 hidden layer with 2 neurons. 
Epoch: 0, Loss: 4.5312e+10, Time Elapsed: 52 milliseconds
Epoch: 1, Loss: 2.8249e+10, Time Elapsed: 46 milliseconds
Epoch: 2, Loss: 2.14156e+10, Time Elapsed: 51 milliseconds
Epoch: 3, Loss: 1.64771e+10, Time Elapsed: 46 milliseconds
Epoch: 4, Loss: 1.27868e+10, Time Elapsed: 49 milliseconds
Epoch: 5, Loss: 1.0011e+10, Time Elapsed: 46 milliseconds
....
Epoch: 994, Loss: 4.34173, Time Elapsed: 46 milliseconds
Epoch: 995, Loss: 4.34072, Time Elapsed: 46 milliseconds
Epoch: 996, Loss: 4.33972, Time Elapsed: 46 milliseconds
Epoch: 997, Loss: 4.33871, Time Elapsed: 56 milliseconds
Epoch: 998, Loss: 4.33771, Time Elapsed: 45 milliseconds
Epoch: 999, Loss: 4.3367, Time Elapsed: 46 milliseconds
```

The third model's convergence to zero is not directly observed. But from the loss appears to drop at a constant speed after it drops below 1000, so we can infer it will eventually reach 0:

```
The third linear model has 2 hidden layers, each having 2 neurons.
Epoch: 0, Loss: 2.43897e+10, Time Elapsed: 69 milliseconds
Epoch: 1, Loss: 1.67572e+10, Time Elapsed: 66 milliseconds
Epoch: 2, Loss: 1.46046e+10, Time Elapsed: 65 milliseconds
Epoch: 3, Loss: 1.27807e+10, Time Elapsed: 70 milliseconds
Epoch: 4, Loss: 1.11588e+10, Time Elapsed: 71 milliseconds
Epoch: 5, Loss: 9.71656e+09, Time Elapsed: 65 milliseconds
...
Epoch: 2994, Loss: 389.533, Time Elapsed: 69 milliseconds
Epoch: 2995, Loss: 389.411, Time Elapsed: 68 milliseconds
Epoch: 2996, Loss: 389.289, Time Elapsed: 64 milliseconds
Epoch: 2997, Loss: 389.168, Time Elapsed: 66 milliseconds
Epoch: 2998, Loss: 389.046, Time Elapsed: 66 milliseconds
Epoch: 2999, Loss: 388.924, Time Elapsed: 67 milliseconds
```

The above test results are stored in `repo/IntegrationTestResults`. To replicate the tests, simply run the `main.cpp` as is. The integrated tests are already uncommented.

The fact that our framework optimizes linear model correctly shows that the __logic of gradient descent is correctly implemented__, at least for linear models. As our implementation of gradient descent __doesn't differentiate between linear or non-linear models__ (in any model, the gradients are calculated numerically using forward perturbance) we should expect the same logic to be correctly applied to non-linear models.

Regarding numerical stability: we did some simple tests on the stability of numerical methods in the unit test sections for functions invoked in `calcJacobians()`. The errors are confined to 0.00001, 10 times the default perturbance.

### Efficiency Test

#### Test minibatch size's impact on training speed 

Effect of minibatch_size on training speed.
The training data here is linearly generated (see code from `DataGeneration.ipynb`). It contains 3500 samples and `X` contains 7 features.
We train a simple regression model (1 `HiddenLayer` with 1 neuron, identity activation) on the data, using sgd with `batch_sizes` = 250, 500, 750, 1000, 1250, 1500, 1750 and 2000. The result is as follows:

```
Batch_size = 250:
Epoch: 0, Loss: 2.53327e+16, Time Elapsed: 327 milliseconds
Epoch: 1, Loss: 2.36613e+16, Time Elapsed: 328 milliseconds
Epoch: 2, Loss: 2.14247e+16, Time Elapsed: 339 milliseconds
Epoch: 3, Loss: 2.13626e+16, Time Elapsed: 337 milliseconds
Epoch: 4, Loss: 1.94872e+16, Time Elapsed: 334 milliseconds
Epoch: 5, Loss: 1.7922e+16, Time Elapsed: 327 milliseconds
Epoch: 6, Loss: 1.71324e+16, Time Elapsed: 339 milliseconds
Epoch: 7, Loss: 1.74064e+16, Time Elapsed: 333 milliseconds
Epoch: 8, Loss: 1.71282e+16, Time Elapsed: 338 milliseconds
Epoch: 9, Loss: 1.63949e+16, Time Elapsed: 351 milliseconds

Batch_size = 500:
Epoch: 0, Loss: 1.50634e+16, Time Elapsed: 482 milliseconds
Epoch: 1, Loss: 1.46926e+16, Time Elapsed: 495 milliseconds
Epoch: 2, Loss: 1.37443e+16, Time Elapsed: 464 milliseconds
Epoch: 3, Loss: 1.30938e+16, Time Elapsed: 473 milliseconds
Epoch: 4, Loss: 1.28944e+16, Time Elapsed: 464 milliseconds
Epoch: 5, Loss: 1.29249e+16, Time Elapsed: 461 milliseconds
Epoch: 6, Loss: 1.1807e+16, Time Elapsed: 480 milliseconds
Epoch: 7, Loss: 1.08319e+16, Time Elapsed: 456 milliseconds
Epoch: 8, Loss: 1.03731e+16, Time Elapsed: 466 milliseconds
Epoch: 9, Loss: 9.72618e+15, Time Elapsed: 483 milliseconds

Batch_size = 750:
Epoch: 0, Loss: 9.19267e+15, Time Elapsed: 593 milliseconds
Epoch: 1, Loss: 9.11161e+15, Time Elapsed: 604 milliseconds
Epoch: 2, Loss: 8.93699e+15, Time Elapsed: 591 milliseconds
Epoch: 3, Loss: 9.08605e+15, Time Elapsed: 599 milliseconds
Epoch: 4, Loss: 9.18936e+15, Time Elapsed: 602 milliseconds
Epoch: 5, Loss: 8.9379e+15, Time Elapsed: 605 milliseconds
Epoch: 6, Loss: 8.84316e+15, Time Elapsed: 608 milliseconds
Epoch: 7, Loss: 8.35956e+15, Time Elapsed: 593 milliseconds
Epoch: 8, Loss: 8.41495e+15, Time Elapsed: 592 milliseconds
Epoch: 9, Loss: 8.41763e+15, Time Elapsed: 607 milliseconds

Batch_size = 1000:
Epoch: 0, Loss: 8.44139e+15, Time Elapsed: 724 milliseconds
Epoch: 1, Loss: 8.09404e+15, Time Elapsed: 739 milliseconds
Epoch: 2, Loss: 7.62665e+15, Time Elapsed: 736 milliseconds
Epoch: 3, Loss: 7.38243e+15, Time Elapsed: 747 milliseconds
Epoch: 4, Loss: 7.07441e+15, Time Elapsed: 740 milliseconds
Epoch: 5, Loss: 6.6707e+15, Time Elapsed: 730 milliseconds
Epoch: 6, Loss: 6.66914e+15, Time Elapsed: 740 milliseconds
Epoch: 7, Loss: 6.53266e+15, Time Elapsed: 736 milliseconds
Epoch: 8, Loss: 6.25586e+15, Time Elapsed: 729 milliseconds
Epoch: 9, Loss: 5.93687e+15, Time Elapsed: 750 milliseconds

Batch_size = 1250:
Epoch: 0, Loss: 5.74441e+15, Time Elapsed: 889 milliseconds
Epoch: 1, Loss: 5.62976e+15, Time Elapsed: 863 milliseconds
Epoch: 2, Loss: 5.40353e+15, Time Elapsed: 869 milliseconds
Epoch: 3, Loss: 5.34083e+15, Time Elapsed: 869 milliseconds
Epoch: 4, Loss: 5.16191e+15, Time Elapsed: 862 milliseconds
Epoch: 5, Loss: 5.23598e+15, Time Elapsed: 884 milliseconds
Epoch: 6, Loss: 5.10366e+15, Time Elapsed: 869 milliseconds
Epoch: 7, Loss: 5.06938e+15, Time Elapsed: 876 milliseconds
Epoch: 8, Loss: 4.99835e+15, Time Elapsed: 867 milliseconds
Epoch: 9, Loss: 5.07655e+15, Time Elapsed: 858 milliseconds

Batch_size = 1500:
Epoch: 0, Loss: 4.95183e+15, Time Elapsed: 1005 milliseconds
Epoch: 1, Loss: 4.82478e+15, Time Elapsed: 1029 milliseconds
Epoch: 2, Loss: 4.82007e+15, Time Elapsed: 987 milliseconds
Epoch: 3, Loss: 4.80423e+15, Time Elapsed: 1002 milliseconds
Epoch: 4, Loss: 4.87371e+15, Time Elapsed: 1006 milliseconds
Epoch: 5, Loss: 4.84422e+15, Time Elapsed: 992 milliseconds
Epoch: 6, Loss: 4.83176e+15, Time Elapsed: 1015 milliseconds
Epoch: 7, Loss: 4.70363e+15, Time Elapsed: 1015 milliseconds
Epoch: 8, Loss: 4.60853e+15, Time Elapsed: 992 milliseconds
Epoch: 9, Loss: 4.59607e+15, Time Elapsed: 995 milliseconds

Batch_size = 1750:
Epoch: 0, Loss: 4.64499e+15, Time Elapsed: 1159 milliseconds
Epoch: 1, Loss: 4.56626e+15, Time Elapsed: 1137 milliseconds
Epoch: 2, Loss: 4.50204e+15, Time Elapsed: 1135 milliseconds
Epoch: 3, Loss: 4.54712e+15, Time Elapsed: 1137 milliseconds
Epoch: 4, Loss: 4.57265e+15, Time Elapsed: 1129 milliseconds
Epoch: 5, Loss: 4.59123e+15, Time Elapsed: 1146 milliseconds
Epoch: 6, Loss: 4.61646e+15, Time Elapsed: 1169 milliseconds
Epoch: 7, Loss: 4.65208e+15, Time Elapsed: 1142 milliseconds
Epoch: 8, Loss: 4.70754e+15, Time Elapsed: 1137 milliseconds
Epoch: 9, Loss: 4.58688e+15, Time Elapsed: 1137 milliseconds

Batch_size = 2000:
Epoch: 0, Loss: 4.58329e+15, Time Elapsed: 1319 milliseconds
Epoch: 1, Loss: 4.49471e+15, Time Elapsed: 1321 milliseconds
Epoch: 2, Loss: 4.45113e+15, Time Elapsed: 1272 milliseconds
Epoch: 3, Loss: 4.42429e+15, Time Elapsed: 1269 milliseconds
Epoch: 4, Loss: 4.43087e+15, Time Elapsed: 1273 milliseconds
Epoch: 5, Loss: 4.43919e+15, Time Elapsed: 1269 milliseconds
Epoch: 6, Loss: 4.47988e+15, Time Elapsed: 1260 milliseconds
Epoch: 7, Loss: 4.41772e+15, Time Elapsed: 1280 milliseconds
Epoch: 8, Loss: 4.42223e+15,
Time Elapsed: 1278 milliseconds
Epoch: 9, Loss: 4.4128e+15, Time Elapsed: 1269 milliseconds
```

It's obvious that the __training time grows linearly with mini batch size__.

#### Test how the number of layers affect training speed

The first model has 4 hidden layers, each having 2 neurons, all activated by sigmoid. Its training performance:

```
Epoch: 0, Loss: 5.81184e+16, Time Elapsed: 1236 milliseconds
Epoch: 1, Loss: 5.81184e+16, Time Elapsed: 1222 milliseconds
Epoch: 2, Loss: 5.81184e+16, Time Elapsed: 1241 milliseconds
Epoch: 3, Loss: 5.81184e+16, Time Elapsed: 1214 milliseconds
Epoch: 4, Loss: 5.81184e+16, Time Elapsed: 1231 milliseconds
Epoch: 5, Loss: 5.81184e+16, Time Elapsed: 1314 milliseconds
Epoch: 6, Loss: 5.81184e+16, Time Elapsed: 1238 milliseconds
Epoch: 7, Loss: 5.81184e+16, Time Elapsed: 1202 milliseconds
Epoch: 8, Loss: 5.81184e+16, Time Elapsed: 1215 milliseconds
Epoch: 9, Loss: 5.81184e+16, Time Elapsed: 1214 milliseconds
```

In contrast, the second model has 6 hidden layers, each with 2 neurons, activated by sigmoid. It has the following training performance:

```
Epoch: 0, Loss: 5.81184e+16, Time Elapsed: 1672 milliseconds
Epoch: 1, Loss: 5.81184e+16, Time Elapsed: 1637 milliseconds
Epoch: 2, Loss: 5.81184e+16, Time Elapsed: 1667 milliseconds
Epoch: 3, Loss: 5.81184e+16, Time Elapsed: 1641 milliseconds
Epoch: 4, Loss: 5.81184e+16, Time Elapsed: 1643 milliseconds
Epoch: 5, Loss: 5.81184e+16, Time Elapsed: 1658 milliseconds
Epoch: 6, Loss: 5.81184e+16, Time Elapsed: 1679 milliseconds
Epoch: 7, Loss: 5.81184e+16, Time Elapsed: 1647 milliseconds
Epoch: 8, Loss: 5.81184e+16, Time Elapsed: 1659 milliseconds
Epoch: 9, Loss: 5.81184e+16, Time Elapsed: 1649 milliseconds
```

A third model has 8 hidden layers, each with 2 neurons, activated by sigmoid. It has the following training performance:

```
Epoch: 0, Loss: 5.81184e+16, Time Elapsed: 2128 milliseconds
Epoch: 1, Loss: 5.81184e+16, Time Elapsed: 2096 milliseconds
Epoch: 2, Loss: 5.81184e+16, Time Elapsed: 2076 milliseconds
Epoch: 3, Loss: 5.81184e+16, Time Elapsed: 2115 milliseconds
Epoch: 4, Loss: 5.81184e+16, Time Elapsed: 2083 milliseconds
Epoch: 5, Loss: 5.81184e+16, Time Elapsed: 2108 milliseconds
Epoch: 6, Loss: 5.81184e+16, Time Elapsed: 2162 milliseconds
Epoch: 7, Loss: 5.81184e+16, Time Elapsed: 2104 milliseconds
Epoch: 8, Loss: 5.81184e+16, Time Elapsed: 2076 milliseconds
Epoch: 9, Loss: 5.81184e+16, Time Elapsed: 2086 milliseconds
```

__Training time grows linearly with number of hidden layers__.

Note the fact that these the loss doesn't converge doesn't affect its training time cost. In each epoch, all gradients are recalculated.



#### Test how number of neurons affect training speed

A model with 1 hidden layer that has 1 neuron has the following training performance:

```
Epoch: 0, Loss: 2.37895e+16, Time Elapsed: 328 milliseconds
Epoch: 1, Loss: 2.18044e+16, Time Elapsed: 331 milliseconds
Epoch: 2, Loss: 2.1399e+16, Time Elapsed: 322 milliseconds
Epoch: 3, Loss: 1.95866e+16, Time Elapsed: 338 milliseconds
Epoch: 4, Loss: 1.78183e+16, Time Elapsed: 334 milliseconds
Epoch: 5, Loss: 1.6501e+16, Time Elapsed: 333 milliseconds
Epoch: 6, Loss: 1.63649e+16, Time Elapsed: 326 milliseconds
Epoch: 7, Loss: 1.5181e+16, Time Elapsed: 359 milliseconds
Epoch: 8, Loss: 1.44907e+16, Time Elapsed: 340 milliseconds
Epoch: 9, Loss: 1.46814e+16, Time Elapsed: 325 milliseconds
```

A model with 1 hidden layer that has 3 neurons has the following training performance:

```
Epoch: 0, Loss: 4.68939e+15, Time Elapsed: 959 milliseconds
Epoch: 1, Loss: 5.03447e+15, Time Elapsed: 945 milliseconds
Epoch: 2, Loss: 4.53709e+15, Time Elapsed: 950 milliseconds
Epoch: 3, Loss: 4.24823e+15, Time Elapsed: 955 milliseconds
Epoch: 4, Loss: 3.95887e+15, Time Elapsed: 948 milliseconds
Epoch: 5, Loss: 3.87876e+15, Time Elapsed: 952 milliseconds
Epoch: 6, Loss: 3.95953e+15, Time Elapsed: 958 milliseconds
Epoch: 7, Loss: 3.81723e+15, Time Elapsed: 968 milliseconds
Epoch: 8, Loss: 3.9007e+15, Time Elapsed: 962 milliseconds
Epoch: 9, Loss: 3.80634e+15, Time Elapsed: 979 milliseconds
```

A model with 1 hidden layer that has 5 neurons has the following training performance:

```
Epoch: 0, Loss: 5.66341e+15, Time Elapsed: 2288 milliseconds
Epoch: 1, Loss: 4.96649e+15, Time Elapsed: 2318 milliseconds
Epoch: 2, Loss: 4.2797e+15, Time Elapsed: 2329 milliseconds
Epoch: 3, Loss: 4.42179e+15, Time Elapsed: 2298 milliseconds
Epoch: 4, Loss: 4.05456e+15, Time Elapsed: 2285 milliseconds
Epoch: 5, Loss: 3.73989e+15, Time Elapsed: 2321 milliseconds
Epoch: 6, Loss: 3.75596e+15, Time Elapsed: 2295 milliseconds
Epoch: 7, Loss: 3.71505e+15, Time Elapsed: 2307 milliseconds
Epoch: 8, Loss: 3.71669e+15, Time Elapsed: 2288 milliseconds
Epoch: 9, Loss: 3.71584e+15, Time Elapsed: 2294 milliseconds
```

A model with 1 hidden layer that has 7 neurons has the following training performance:

```
Epoch: 0, Loss: 3.20344e+16, Time Elapsed: 4633 milliseconds
Epoch: 1, Loss: 2.82482e+16, Time Elapsed: 4841 milliseconds
Epoch: 2, Loss: 2.07291e+16, Time Elapsed: 5471 milliseconds
Epoch: 3, Loss: 1.17285e+16, Time Elapsed: 4887 milliseconds
Epoch: 4, Loss: 1.01241e+16, Time Elapsed: 4650 milliseconds
Epoch: 5, Loss: 1.00147e+16, Time Elapsed: 4632 milliseconds
Epoch: 6, Loss: 1.14448e+16, Time Elapsed: 4628 milliseconds
Epoch: 7, Loss: 6.18926e+15, Time Elapsed: 4603 milliseconds
Epoch: 8, Loss: 5.34442e+15, Time Elapsed: 4617 milliseconds
Epoch: 9, Loss: 5.62928e+15, Time Elapsed: 4636 milliseconds
```

__Training time grows exponentially with number of outputs/neurons in a hidden layer__.

## Implementation Highlight

