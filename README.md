## Project Context

In `VanillaMLP` we created a framework that allows users to create and train a basic Multi-Layer Perceptron model in C++. 

This is a practice project to gain an elementary understanding of how a Neural Network is implemented. Our implementation is mostly naïve, and very likely has serious flaws. This mini report aims to clearly describe the key components in our design. By doing so we hope to expose our current level of understanding (or the lack thereof), and invite _directed_ critical feedback. Your kindness in helping us further  learning would be highly appreciated!

## How to run `VanillaMLP`

### Prerequisites

- We used the non-standard `#pragrama once` as include guard. Most popular compilers support this directive. We recommend using any GCC compiler with versions later than 3.4.

- The matrix algebra is implemented through library `Eigen 3`. The easiest way to install `Eigen 3` is through Microsoft's C++ package manager `vcpkg`. Please refer to https://github.com/Microsoft/vcpkg for `vcpkg` installation. 

- After installing `vcpkg`, `cd` to its hosting directory, and run one of the following depending on the shell you're using:

  ```
  PS> .\vcpkg install eigen3
  Linux:~/$ ./vcpkg install sdl2 curl
  ```

### Using `VanillaMLP`

#### Getting Source Files

Clone the source file repo through

```
git clone https://github.com/shiyangni/VanillaMLP.git
```

Once complete, open an C++ IDE of your choice , start an empty project, and add the header and source files according to the IDE





Users can obtain a customized model structure by specifying the number of hidden layers, how many neurons each hidden layer contains, and the activation function for each hidden layer. 

The framework employs two designs to boost the SGD training efficiency:

- Numerical differentiation and caching for intermediate products in back prop

The framework is vanilla due to the following restrictions:

- No regularization or normalization 