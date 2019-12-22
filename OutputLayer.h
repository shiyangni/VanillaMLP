#pragma once
#include "Layer.h"
#include <Eigen/Dense>
/*An outputLayer simply takes the output of the last hidden layer, and adds everything together.*/
class OutputLayer :
	public Layer
{
public:
	OutputLayer(int numberInputs);
	void calcOutput() override;
private:
	/*This function takes in the input vector and returns its dot product with 1 vector. 
	Defined for nuermical differnetiation.*/
	Eigen::VectorXd calcOutputFromInput(Eigen::VectorXd input);
};

