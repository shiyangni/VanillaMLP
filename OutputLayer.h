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
	/*Calculates and returns do/dinput. The dimension is numInputs X 1.*/
	Eigen::MatrixXd calcDoDinput(double perturbance=0.000001);
private:

};

