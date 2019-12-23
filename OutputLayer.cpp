#include <Eigen/Dense>
#include "OutputLayer.h"

using namespace Eigen;

OutputLayer::OutputLayer()
{
	Layer(0, 0);
}

OutputLayer::OutputLayer(int numberInputs)
{
	Layer(numberInputs, numberInputs);
	setNumInputs(numberInputs);
	setNumOutputs(1);
}

void OutputLayer::calcOutput()
{
	RowVectorXd ones = RowVectorXd::Ones(getNumInputs());
	VectorXd result = ones * getInput();
	setOutput(result);
}

Eigen::MatrixXd OutputLayer::calcDoDinput(double perturbance)
{
	MatrixXd result(getNumInputs(), getNumOutputs());
	RowVectorXd ones = RowVectorXd::Ones(getNumInputs());
	for (int i = 0; i < getNumInputs(); i++) {
		VectorXd perturbedInput = getInput();
		perturbedInput(i) += perturbance;
		VectorXd perturbedOutput = ones * perturbedInput;
		result(i, 0) = (perturbedOutput(0) - getOutput()(0)) / perturbance;
	}
	currSample_DoDinput = result;
	return result;
}

Eigen::MatrixXd& OutputLayer::getCurrSample_DoDinput()
{
	return currSample_DoDinput;
}


