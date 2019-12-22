#include <Eigen/Dense>
#include "HiddenLayer.h"
#include "Layer.h"
#include "Utilities.h"

using namespace std;
using namespace Eigen;

HiddenLayer::HiddenLayer(int numberInputs, int numberOutputs)
{
	Layer(numberInputs, numberOutputs);
	setNumInputs(numberInputs);
	setNumOutputs(numberOutputs);
	/*By default all weights and biases are initialized to one.*/
	setWeights(MatrixXd::Ones(getNumOutputs(), getNumInputs()));
	setBias(VectorXd::Ones(getNumOutputs()));
	setActivation(bentIdentity);
}


Eigen::VectorXd HiddenLayer::calcOutputFromInput(Eigen::VectorXd input)
{
	VectorXd result = activate_vector(getWeights() * input + getBias());
	return result;
}



Eigen::VectorXd HiddenLayer::calcOutputFromBias(Eigen::VectorXd bias)
{
	VectorXd result = activate_vector(getWeights() * getInput() + bias);
	return Eigen::VectorXd();
}


void HiddenLayer::calcOutput()
{
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
	return Eigen::VectorXd();
}

void HiddenLayer::setBias(Eigen::VectorXd newBias)
{
	bias = newBias;
}

Eigen::VectorXd HiddenLayer::getCurrSample_ChainRuleFactor()
{
	return Eigen::VectorXd();
}

void HiddenLayer::setCurrSample_ChainRuleFactor(Eigen::VectorXd)
{
}

Eigen::MatrixXd HiddenLayer::getCurrSample_dodinput()
{
	return Eigen::MatrixXd();
}

void HiddenLayer::setCurrSample_dodinput(Eigen::MatrixXd)
{
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
