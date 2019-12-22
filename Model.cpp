#include "Model.h"
#include "Layer.h"
#include "InputLayer.h"
#include "Utilities.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

void Model::addInputLayer()
{
	int numFeatures = X_train.cols();
	InputLayer inputLayer(numFeatures);
	layers.push_back(inputLayer);
}

Model::Model(Eigen::MatrixXd data)
{
	data_train = data;
	X_train = data.block(0, 1, data.rows(), data.cols() - 1);
	y_train = data.col(0);
	lambda = 0.0005;
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
}




Layer& Model::getLayer(int i)
{
	
	return layers.at(i);
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


