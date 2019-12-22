#pragma once
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <istream>
#include <vector>
#include <string>  

#include "Layer.h"
#include "Utilities.h"



/* Here's how you use a model. 
	- mlp = Model(data).  
	- Then add structure. 
	- mlp.train(lambda = , minibatch_size = , epochs = )
	- mlp.predict(new_X).

 In the first step, data is stored as a Eigen::MatrixXd, where the first column is the 
 label, and the rest are features. The data type restricts our input to numerical values.

 */

class Model
{
private:
	Eigen::MatrixXd X_train;
	Eigen::VectorXd y_train;
	Eigen::MatrixXd data_train;
	std::vector<Layer> layers;
	double lambda;


	/*Does 1 iteration of gradient descent on X and y. Utilizes forwardProp,
	loss, and backProp.*/
	void gd_oneIteration(Eigen::MatrixXd X, Eigen::VectorXd y, double lambda);

	/*This is to be used upon the initialization of model.*/
	void addInputLayer();
public:
	/*The constructor reads in the data and stores them in X_train and y_train. 
	It also initializes the input layer. 
	Assume the first column of data is the label, and the rest are features.*/
	Model(Eigen::MatrixXd data);

	/*The default constructor inits a model based off some random data.*/
	Model();

	/*Add a hidden layer with specified number of outputs. The added layer's input
	is retrieved from the last layer, and the output of that layer is computed and 
	stored.*/
	void addHiddenLayer(int numOutputs);


	/*Add an output layer. Upon adding, the layer's input is retrieved retrieved 
	from the last layer, and is computed and stored.*/
	void addOutputLayer();

	/*Computes y_hat on a given X. X has to have the same number of columns as
	X_train. The returned y_hat has to have the same number of rows as X. */
	Eigen::VectorXd forwardProp(Eigen::MatrixXd X);

	/*Computes the mse loss of y_hat based on X against y. Uses forward_prop. */
	double loss(Eigen::MatrixXd X, Eigen::VectorXd y);

    /*Computes the gradient of loss on weights and bias of all hidden layers, 
	and stores them in the associated layers. Utilizes forwardProp and loss.*/
	void backProp(Eigen::MatrixXd X, Eigen::VectorXd y);

	/*Updates parameter using SGD on X_train and y_train. Utilizes gd_oneIteration.*/
	void train(double lambda);

	Layer& getLayer(int i);

	Eigen::MatrixXd& getDataTrain();

	Eigen::MatrixXd& getXTrain();

	Eigen::VectorXd& getYTrain();

};
 










