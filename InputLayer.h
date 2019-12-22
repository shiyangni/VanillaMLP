#pragma once

#include <Eigen/Dense>
#include "Layer.h"

/*Within a model, the InputLayer is initiated when the model is initiated. 

An input layer always has number of outputs equaling the number of inputs.

- readInput: it only reads input from a specfied source wrapped in a Eigen::VectorXd.
- calcOutput: simply copies and assigns inputVector to outputVector.*/
class InputLayer :
	public Layer
{
public:
	InputLayer(int numberInputs);
	void calcOutput() override;

};

