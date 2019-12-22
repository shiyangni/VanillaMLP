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

HiddenLayer::HiddenLayer(int numberInputs, int numberOutputs, double activate(double))
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

Eigen::MatrixXd HiddenLayer::getCurrSample_dodinput()
{
	return currSample_dodinput;
}

void HiddenLayer::setCurrSample_dodinput(Eigen::MatrixXd newValue)
{
	currSample_dodinput = newValue;
}


void HiddenLayer::addCurrSample_dodweights()
{
	
}

std::vector<Eigen::MatrixXd> HiddenLayer::getCurrSample_dodweights()
{
	return std::vector<Eigen::MatrixXd>();
}

void HiddenLayer::addCurrSample_dodbias()
{
}

std::vector<Eigen::MatrixXd> HiddenLayer::getCurrSample_dodbias()
{
	return std::vector<Eigen::MatrixXd>();
}

Eigen::VectorXd HiddenLayer::getNeblaWeights()
{
	return Eigen::VectorXd();
}



Eigen::VectorXd HiddenLayer::getNeblaBias()
{
	return Eigen::VectorXd();
}



void HiddenLayer::setActivation(double func(double))
{
	activate_scalar = func;
}
