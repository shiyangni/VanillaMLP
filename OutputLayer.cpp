#include <Eigen/Dense>
#include "OutputLayer.h"

using namespace Eigen;

OutputLayer::OutputLayer(int numberInputs)
{
	Layer(numberInputs, numberInputs);
	setNumInputs(numberInputs);
	setNumOutputs(1);
}

void OutputLayer::calcOutput()
{
	VectorXd result = calcOutputFromInput(getInput());
	setOutput(result);
}

Eigen::VectorXd OutputLayer::calcOutputFromInput(VectorXd input)
{
	int numRows = input.rows();
	VectorXd ones(numRows);
	ones.fill(1);
	VectorXd result = ones.transpose() * input;
	return result;
}
