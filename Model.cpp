#include "Model.h"
#include "Layer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"
#include "Utilities.h"
#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <chrono>


using namespace Eigen;
using namespace std;



Model::Model(Eigen::MatrixXd data)
{
	data_train = data;
	X_train = data.block(0, 1, data.rows(), data.cols() - 1);
	y_train = data.col(0);
	lambda = 0.0005;
	numOfLayers = 1;
	addInputLayer();
}

/*An empty model should never be instantiated. It's only written to make sure a default 
case exists.*/
Model::Model()
{
	MatrixXd data_fake;
	data_fake = readCSV("data_fake.csv");
	data_train = data_fake;
	X_train = data_fake.block(0, 1, data_fake.rows(), data_fake.cols() - 1);
	y_train = data_fake.col(0);
	lambda = 0.0005;
	numOfLayers = 1;
	
}

void Model::addInputLayer()
{
	int numFeatures = X_train.cols();
	int layerIndex = 0;
	InputLayer currInputLayer(numFeatures);
	inputLayer = currInputLayer;
	inputLayer.setLayerIndex(0);
	numOfLayers += 1;
}

void Model::addHiddenLayer(int numOutputs, std::function<double(double)> activate)
{
	int currLayerIndex = 1 + hiddenLayers.size();
	int numInputs = getLayer(currLayerIndex - 1).getNumOutputs();
	HiddenLayer hiddenLayer(numInputs, numOutputs, activate);
	hiddenLayer.setLayerIndex(currLayerIndex);
	hiddenLayers.push_back(hiddenLayer);
	numOfLayers += 1;
}

void Model::addOutputLayer()
{
	int currlayerIndex = 1 + hiddenLayers.size();
	int numInputs = getLayer(currlayerIndex - 1).getNumOutputs();
	OutputLayer currOutputLayer(numInputs);
	currOutputLayer.setLayerIndex(currlayerIndex);
	outputLayer = currOutputLayer;
}

double Model::currSample_forwardProp(Eigen::VectorXd x)
{
	inputLayer.readInput(x);
	inputLayer.calcOutput();
	for (int i = 1; i < numOfLayers; i++) {
		VectorXd outputFromLastLayer = getLayer(i - 1).getOutput();
		getLayer(i).readInput(outputFromLastLayer);
		getLayer(i).calcOutput();
		//cout << "The output at layer " << i << " is : \n" << getLayer(i).getOutput() << endl;
	}
	return outputLayer.getOutput()(0);
}

Eigen::VectorXd Model::forwardProp(Eigen::MatrixXd data)
{
	int n = data.rows();
	int q = data.cols() - 1;
	VectorXd y = data.col(0);
	MatrixXd X = data.block(0, 1, n, q);
	VectorXd y_hat = y;
	for (int i = 0; i < n; i++) {
		VectorXd curr_x = X.row(i).transpose();
		y_hat(i) = currSample_forwardProp(curr_x);
	}
	return y_hat;
}

Eigen::VectorXd Model::predict(Eigen::MatrixXd newData)
{
	return forwardProp(newData);
}

void Model::currSample_updateJacobians()
{
	for (HiddenLayer& hl : hiddenLayers) {
		hl.currSample_calcJacobians();
	}
	outputLayer.calcDoDinput();
}

void Model::backProp(Eigen::MatrixXd data)
{
	int n = data.rows();
	int q = data.cols() - 1;
	VectorXd y = data.col(0);
	MatrixXd X = data.block(0, 1, n, q);
	for (int i = 0; i < n; i++) {
		VectorXd curr_x = X.row(i).transpose();
		double curr_y = y(i);
		currSample_forwardProp(curr_x);
		currSample_backProp(curr_y);
	}
	//cout << "The current bySample_neblaWeights size for the first HiddenLayer "
	//	"should be " << n << ". It actually is: \n" 
	//	<< getKthHiddenLayer(0).getNeblaWeights_BySampleVector().size() << endl;
	calcNeblas();
}

void Model::currSample_backProp(double y)
{
	currSample_updateJacobians();
	currSample_updateChainRuleFactors(y);
	currSample_addBySampleNeblas();
}

double Model::mseLoss(Eigen::VectorXd y_hat, Eigen::VectorXd y)
{
	int n = y_hat.rows();
	VectorXd diff = y_hat - y;
	return pow(diff.squaredNorm(), 2) / n;
}

void Model::train_gd(Eigen::MatrixXd data, int epochs, double lambda)
{
	for (int i = 0; i < epochs; i++) {
		auto start = std::chrono::high_resolution_clock::now();
		backProp(data);
		updateParams(lambda);
		VectorXd y_hat = forwardProp(data);
		double loss = mseLoss(y_hat, data.col(0));
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		cout << "Epoch: " << i << ", Loss: " << loss << ", Time Elapsed: " 
			<< microseconds/1000 << " milliseconds" << endl;
	}
}

void Model::train_sgd(Eigen::MatrixXd data, int epochs, double lambda, int batch_size)
{
	int n = data.rows();
	int maximalStartIndex = n - batch_size;
	for (int i = 0; i < epochs; i++) {
		int startIndex = rand() % maximalStartIndex; 
		MatrixXd data_minibatch = data.block(startIndex, 0, batch_size, data.cols());
		auto start = std::chrono::high_resolution_clock::now();
		backProp(data_minibatch);
		updateParams(lambda);
		VectorXd y_hat = forwardProp(data);
		double loss = mseLoss(y_hat, data.col(0));
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		cout << "Epoch: " << i << ", Loss: " << loss << ", Time Elapsed: "
			<< microseconds/1000 << " milliseconds" << endl;
	}
}


double Model::calcCurrSample_DlossDoFinal(double y_true, double perturbance)
{
	double oFinal = getOutputLayer().getOutput()(0);
	double originalLoss = oneSample_MSEloss(y_true, oFinal);
	double perturbedOFinal = oFinal + perturbance;
	double perturbedLoss = oneSample_MSEloss(y_true, perturbedOFinal);
	double result = (perturbedLoss - originalLoss) / perturbance;
	currSample_DlossDoFinal = result;
	return result;
}

double Model::getCurrSample_DlossDoFinal()
{
	return currSample_DlossDoFinal;
}

double Model::oneSample_MSEloss(double y_true, double y_hat)
{
	return pow((y_hat - y_true),2);
}

void Model::currSample_updateChainRuleFactors()
{
	int k = hiddenLayers.size() - 1;
	MatrixXd runningChainRuleFactor 
		= getOutputLayer().getCurrSample_DoDinput() * getCurrSample_DlossDoFinal();
	getKthHiddenLayer(k).setCurrSample_ChainRuleFactor(runningChainRuleFactor);
	//cout << "TESTING: The " << k << "-th hidden layer now has runningChainRuleFactor: \n" << runningChainRuleFactor  << endl;
	while (k > 0) {
		runningChainRuleFactor
			= getKthHiddenLayer(k).getCurrSample_DoDinput() * runningChainRuleFactor;
		k--;
		getKthHiddenLayer(k).setCurrSample_ChainRuleFactor(runningChainRuleFactor);
		//cout << "TESTING: The " << k << "-th hidden layer now has runningChainRuleFactor: \n" << runningChainRuleFactor << endl;
	}
}

void Model::currSample_updateChainRuleFactors(double y_true, double perturbance)
{
	calcCurrSample_DlossDoFinal(y_true, perturbance);
	currSample_updateChainRuleFactors();
}

void Model::currSample_addBySampleNeblas()
{
	for (HiddenLayer& hl : hiddenLayers) {
		hl.addCurrSample_neblas();
	}
}

void Model::calcNeblas()
{
	for (HiddenLayer& hl : hiddenLayers) {
		hl.calcNeblas();
	}
}

void Model::updateParams(double lambda)
{
	for (HiddenLayer& hl : hiddenLayers) {
		hl.updateParams(lambda);
	}
}

Layer& Model::getLayer(int i)
{
	if (i == 0) {
		return getInputLayer();
	}
	else if (i < numOfLayers - 1) {
		return getKthHiddenLayer(i - 1);
	}
	else {
		return getOutputLayer();
	}
}

InputLayer& Model::getInputLayer()
{
	return inputLayer;
}

HiddenLayer& Model::getKthHiddenLayer(int k)
{
	return hiddenLayers.at(k);
}

OutputLayer& Model::getOutputLayer()
{
	return outputLayer;
}

Eigen::MatrixXd& Model::getDataTrain()
{
	return data_train;
}

Eigen::MatrixXd& Model::getXTrain()
{
	return X_train;
}

Eigen::VectorXd& Model::getYTrain()
{
	return y_train;
}

int Model::getNumOfLayers()
{
	return numOfLayers;
}


