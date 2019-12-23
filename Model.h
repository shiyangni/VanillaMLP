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
#include "HiddenLayer.h"
#include "InputLayer.h"
#include "OutputLayer.h"
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
	InputLayer inputLayer;
	std::vector<HiddenLayer> hiddenLayers;
	OutputLayer outputLayer;
	double lambda;
	double currSample_DlossDoFinal;
	/*Always includes the input and the output layer. Thus at any time the minimum
	value is 2. use 1 + hiddenLayers.size() to get the index of the to-be added layer.*/
	int numOfLayers;


	/*Does 1 iteration of gradient descent on X and y. Utilizes forwardProp,
	loss, and backProp.*/
	void gd_oneIteration(Eigen::MatrixXd X, Eigen::VectorXd y, double lambda);

	/*This is used upon the initialization of model.*/
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
	void addHiddenLayer(int numOutputs, std::function<double(double)> activate=bentIdentity);

	/*Add an output layer. Upon adding, the layer's input is retrieved retrieved 
	from the last layer, and is computed and stored.*/
	void addOutputLayer();
	
	/*Computes y_hat on one given sample. 
	Lets all layers readInput and calcOutput on the given sample.*/
	double currSample_forwardProp(Eigen::VectorXd x);

	/*Computes y_hat on a given X. X has to have the same number of columns as
	X_train. The returned y_hat has to have the same number of rows as X. */
	Eigen::VectorXd forwardProp(Eigen::MatrixXd data);

	/*Computes the gradient of loss on weights and bias of all hidden layers,
	and stores them in the associated layers. Utilizes forwardProp and loss.*/
	void backProp(Eigen::MatrixXd data);

	/*Updates all Jacobians(i.e., do/dinput, all do/dweightJ, and all do/dbiasJ) 
	for all layers on the current sample. 
	ONLY INVOKED after currSample_forwardProp();*/
	void currSample_updateJacobians();



	/*Computes and adds all neblas on the current sample, described by x and y. Note:
	- currSample_backProp doesn't invoke currSample_forwardProp. So the sample x needs
	to be read in manually. This is just so that the design pattern is clearer.
	- x is a column vector. So we need to transpose the rows from the dataset.
	- backProp is not just repeating currSample_backProp on all rows! It also aggregates
	over the cached per sample neblas to calculate data-wise neblas, and clears those caches!*/
	void currSample_backProp(double y);

	/*Computes the mse loss of y_hat against y. */
	double mseLoss(Eigen::VectorXd y_hat, Eigen::VectorXd y);

	/*Trains the model on data using Gradient Descent.*/
	void train_gd(Eigen::MatrixXd data_train, int epochs, double lambda = 0.005);

	/*Calculates the right-most element in the currSample_ChainRuleFactor and
	caches the output in the field. oFinal is just another name for y_hat on the 
	current sample.*/
	double calcCurrSample_DlossDoFinal(double y_true, double perturbance = 0.000001);

	double getCurrSample_DlossDoFinal();

	double oneSample_MSEloss(double y_true, double y_hat);

	/*The function that updates chainRuleFactors in all hidden layers. 
	The no-argument version should only be invoked after running calcCurrSample_DlossDoFinal.*/
	void currSample_updateChainRuleFactors();

	/*A wrapper function that updates the ChianRuleFactors according to a true label.*/
	void currSample_updateChainRuleFactors(double y_true, double perturbance = 0.000001);

	/*Lets all hidden layers add the currentSample_neblaWeights and currSample_neblaBias
	to the caching vectors. Invoked ONLY after currSample_forwardProp, currSample_calcJacobians, 
	and currSample_updateChainRuleFactors(). */
	void currSample_addBySampleNeblas();

	/*Lets all the layers calcNeblas. Invoked after doing currSample_forwardProp() 
	and currSample_backwardProp() on all rows of training data.*/
	void calcNeblas();

	/*Lets all hidden layers update weights and biases. The default learning rate is 0.005.*/
	void updateParams(double lambda=0.005);

	Layer& getLayer(int i);

	InputLayer& getInputLayer();

	HiddenLayer& getKthHiddenLayer(int k);

	OutputLayer& getOutputLayer();

	Eigen::MatrixXd& getDataTrain();

	Eigen::MatrixXd& getXTrain();

	Eigen::VectorXd& getYTrain();

	int getNumOfLayers();


};
 










