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

Eigen::MatrixXd HiddenLayer::calcCurrSample_DlossDWeights()
{
	MatrixXd result(getNumInputs(), getNumOutputs());
	for (int i = 0; i < getNumOutputs(); i++) {
		result.col(i) = currSample_DoDweights.at(i) * currSample_chainRuleFactor;
	}
	//cout << "TESTING: The calcualted DlossDweights is : \n" << result << endl;
	return result;
}

void HiddenLayer::addCurrSample_neblaWeights()
{
	MatrixXd currSample_neblaWeights = calcCurrSample_DlossDWeights().transpose();
	neblaWeights_bySample.push_back(currSample_neblaWeights);
	//cout << "TESTING: The pushed in currSample_neblaWeights is: \n" << currSample_neblaWeights << endl;
	currSample_DoDweights.clear();
}

Eigen::MatrixXd HiddenLayer::getJthSample_neblaWeights(int j)
{
	return neblaWeights_bySample.at(j);
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
	//cout << "TESTING: The currSample_DoDinput is :\n" << result << endl;
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

Eigen::MatrixXd& HiddenLayer::getCurrSample_DoDweightJ(int j)
{
	return currSample_DoDweights.at(j);
}

void HiddenLayer::calcCurrSample_DoDbias()
{
	int numOutputs = getNumOutputs();
	for (int j = 0; j < numOutputs; j++) {
		MatrixXd DoDbiasJ = calcDoDbiasJ(j);
		currSample_DoDbias.push_back(DoDbiasJ);
		//cout << "TESTINGL: The " << j << "th currSampleDoDbiasJ is：\n" << DoDbiasJ << endl;
	}
}

std::vector<Eigen::MatrixXd>& HiddenLayer::getCurrSample_DoDbias()
{
	return currSample_DoDbias;
}

Eigen::MatrixXd& HiddenLayer::getCurrSample_DoDbiasJ(int j)
{
	return currSample_DoDbias.at(j);
}

Eigen::RowVectorXd HiddenLayer::calcCurrSample_DlossDbias()
{
	RowVectorXd result(getNumOutputs());
	for (int i = 0; i < getNumOutputs(); i++) {
		result.col(i) = currSample_DoDbias.at(i) * currSample_chainRuleFactor;
	}
	//cout << "TESTING: The calcualted DlossDbias is: \n" << result << endl;
	return result;
}

void HiddenLayer::addCurrSample_neblaBias()
{
	VectorXd currSample_neblaBias = calcCurrSample_DlossDbias().transpose();
	neblaBias_bySample.push_back(currSample_neblaBias);
	//cout << "TESTING: The pushed in currSample_neblaWeights is: \n" << currSample_neblaBias << endl;
	currSample_DoDbias.clear();
}

Eigen::VectorXd HiddenLayer::getJthSample_neblaBias(int j)
{
	return neblaBias_bySample.at(j);
}

void HiddenLayer::addCurrSample_neblas()
{
	addCurrSample_neblaWeights();
	addCurrSample_neblaBias();
}

void HiddenLayer::currSample_calcJacobians()
{
	calcDoDinput();
	calcCurrSample_DoDbias();
	calcCurrSample_DoDweights();
}

Eigen::MatrixXd HiddenLayer::calcNeblaWeights()
{
	MatrixXd result = MatrixXd::Zero(getNumOutputs(), getNumInputs());
	int count = 0;
	for (MatrixXd a : neblaWeights_bySample) {
		result += a;
		count += 1;
	}
	neblaWeights_bySample.clear();
	//cout << "TESTING: For this data, the currSample_neblaWeights sum is: \n" << result << endl;
	neblaWeights = result / count;
	//cout << "TESTING: For this data, the calculated nebla weights is :\n" << neblaWeights << endl;
	return neblaWeights;
}

Eigen::VectorXd HiddenLayer::calcNeblaBias()
{
	VectorXd result = VectorXd::Zero(getNumOutputs());
	int count = 0;
	for (VectorXd a : neblaBias_bySample) {
		result += a;
		count += 1;
	}
	neblaBias_bySample.clear();
	//cout << "TESTING: For this data, the currSample_neblaBias sum is: \n" << result << endl;
	neblaBias = result / count;
	//cout << "TESTING: For this data, the calculated nebla bias is :\n" << neblaBias << endl;
	return neblaBias;
}

void HiddenLayer::calcNeblas()
{
	calcNeblaWeights();
	calcNeblaBias();
}

std::vector<Eigen::MatrixXd>& HiddenLayer::getNeblaWeights_BySampleVector()
{
	return neblaWeights_bySample;
}

std::vector<Eigen::VectorXd>& HiddenLayer::getNeblaBias_BySampleVector()
{
	return neblaBias_bySample;
}

Eigen::MatrixXd& HiddenLayer::getNeblaWeights()
{
	return neblaWeights;
}



Eigen::VectorXd& HiddenLayer::getNeblaBias()
{
	return neblaBias;
}

void HiddenLayer::updateWeights(double learningRate)
{
	MatrixXd currWeights = getWeights();
	setWeights(currWeights - neblaWeights * learningRate);
	
}

void HiddenLayer::updateBias(double learningRate)
{
	VectorXd currBias = getBias();
	setBias(currBias - neblaBias * learningRate);
}

void HiddenLayer::updateParams(double learningRate)
{
	updateWeights(learningRate);
	updateBias(learningRate);
}



void HiddenLayer::setActivation(std::function<double(double)> func)
{
	activate_scalar = func;
}
