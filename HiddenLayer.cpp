#include <Eigen/Dense>
#include "HiddenLayer.h"
#include "Layer.h"
#include "Utilities.h"

using namespace std;
using namespace Eigen;



Eigen::VectorXd HiddenLayer::returnOutput()
{
	VectorXd result = activate_vector(getWeights() * getInput() + getBias());
	return result;
}

HiddenLayer::HiddenLayer()
{
	Layer(0, 0);
}

HiddenLayer::HiddenLayer(int numberInputs, int numberOutputs, std::function<double(double)> activate)
{
	Layer(numberInputs, numberOutputs);
	setNumInputs(numberInputs);
	setNumOutputs(numberOutputs);
	setWeights(MatrixXd::Ones(getNumOutputs(), getNumInputs()));
	setBias(VectorXd::Ones(getNumOutputs()));
	setActivation(activate);
}



Eigen::MatrixXd HiddenLayer::calcDoDweightJ(int j, double perturbance)
{
	MatrixXd weights = getWeights();
	VectorXd bias = getBias();
	VectorXd input = getInput();
	VectorXd originalOutput = activate_vector(weights * input + bias);
	VectorXd weightJ = getJthWeight(j);
	MatrixXd result(weightJ.rows(), originalOutput.rows());
	for (int i = 0; i < originalOutput.rows(); i++) {
		for (int k = 0; k < weightJ.rows(); k++) {
			MatrixXd perturbedWeights = weights;
			perturbedWeights(j, k) += perturbance;
			VectorXd perturbedOutput = activate_vector(perturbedWeights * input + bias);
			result(k, i) = (perturbedOutput(i) - originalOutput(i)) / perturbance;
		}
	}
	return result;
}

Eigen::MatrixXd HiddenLayer::calcDoDbiasJ(int j, double perturbance)
{
	VectorXd bias = getBias();
	VectorXd originalOutput = returnOutput();
	MatrixXd result(1, originalOutput.rows());
	for (int k = 0; k < originalOutput.rows(); k++) {
		VectorXd perturbedBias = bias;
		perturbedBias(j) += perturbance;
		VectorXd perturbedOutput = activate_vector(getWeights() * getInput() + perturbedBias);
		result(0, k) = (perturbedOutput(k) - originalOutput(k)) / perturbance;
	}
	return result;
}


Eigen::MatrixXd HiddenLayer::calcDoDinput(double perturbance)
{
	VectorXd input = getInput();
	VectorXd originalOutput = returnOutput();
	MatrixXd result(input.rows(), originalOutput.rows());
	for (int i = 0; i < input.rows(); i++) {
		for (int j = 0; j < originalOutput.rows(); j++) {
			VectorXd perturbedInput = input;
			perturbedInput(i) += perturbance;
			VectorXd perturbedOutput = activate_vector(getWeights() * perturbedInput + getBias());
			result(i, j) = (perturbedOutput(j) - originalOutput(j)) / perturbance;
		}
	}
	currSample_DoDinput = result;
	return result;
}



void HiddenLayer::calcOutput()
{
	VectorXd output = returnOutput();
	setOutput(output);
}

Eigen::VectorXd HiddenLayer::activate_vector(Eigen::VectorXd input)
{
	VectorXd result = VectorXd::Ones(input.rows());
	for (int i = 0; i < input.rows(); i++) {
		result(i) = activate_scalar(input(i));
	}
	return result;
}

Eigen::MatrixXd HiddenLayer::getWeights()
{
	return weights;
}

void HiddenLayer::setWeights(Eigen::MatrixXd newWeights)
{
	weights = newWeights;
}

Eigen::VectorXd HiddenLayer::getJthWeight(int j)
{
	VectorXd jthWeight = weights.row(j).transpose();
	return jthWeight;
}

Eigen::VectorXd HiddenLayer::getBias()
{
	return bias;
}

void HiddenLayer::setBias(Eigen::VectorXd newBias)
{
	bias = newBias;
}

Eigen::VectorXd HiddenLayer::getCurrSample_ChainRuleFactor()
{
	return currSample_chainRuleFactor;
}

void HiddenLayer::setCurrSample_ChainRuleFactor(Eigen::VectorXd newCRfactor)
{
	currSample_chainRuleFactor = newCRfactor;
}

Eigen::MatrixXd HiddenLayer::getCurrSample_DoDinput()
{
	return currSample_DoDinput;
}



void HiddenLayer::calcCurrSample_DoDweights()
{
	int numOutputs = getNumOutputs();
	for (int j = 0; j < numOutputs; j++) {
		MatrixXd DoDweightJ = calcDoDweightJ(j);
		currSample_DoDweights.push_back(DoDweightJ);
	}
}

std::vector<Eigen::MatrixXd>& HiddenLayer::getCurrSample_DoDweights()
{
	return currSample_DoDweights;
}

void HiddenLayer::calcCurrSample_DoDbias()
{
	int numOutputs = getNumOutputs();
	for (int j = 0; j < numOutputs; j++) {
		MatrixXd DoDbiasJ = calcDoDbiasJ(j);
		currSample_DoDbias.push_back(DoDbiasJ);
	}
}

std::vector<Eigen::MatrixXd>& HiddenLayer::getCurrSample_DoDbias()
{
	return currSample_DoDbias;
}

void HiddenLayer::calcJacobians()
{
	calcDoDinput();
	calcCurrSample_DoDbias();
	calcCurrSample_DoDweights();
}

Eigen::VectorXd HiddenLayer::getNeblaWeights()
{
	return neblaWeights;
}



Eigen::VectorXd HiddenLayer::getNeblaBias()
{
	return Eigen::VectorXd();
}



void HiddenLayer::setActivation(std::function<double(double)> func)
{
	activate_scalar = func;
}
