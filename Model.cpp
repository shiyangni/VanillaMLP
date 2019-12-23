#include "Model.h"
#include "Layer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"
#include "Utilities.h"
#include <Eigen/Dense>

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

double Model::forwardProp_oneSample(Eigen::VectorXd x)
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


