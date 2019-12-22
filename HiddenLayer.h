#pragma once
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include "Layer.h"

/* A hidden layer has 
- a weighting matrix, and the associated derivative matrix. numOutputs X numInputs dimension.
- a bias vector, and the associated derivative matrix, numOutputs X 1 dimension.
- an activation function. We implement it as a sigmoid function. 

The neblas are the final derivatives used in sample_wise gradient descent. This means 
1. It's the averaged derivative dloss/dweights over all rows. These summand derivatives need to be stored somewhere.
2. It's only calculatable from the model. The fields here only act as storage. 


How are the neblas calculated?
For EACH SAMPLE, dloss/dw_kj = do_k/dw_kj * do_(k+1)/do_k * ... * do_L/do_(L-1) * do_final/do_L
(here the subscript k can be omitted since we're operating within one layer)*/
class HiddenLayer :
	public Layer
{
	friend class Model;
private:
	Eigen::MatrixXd weights;
	Eigen::VectorXd bias;
	/*For the k-th hidden layer, this stores the (do_(k+1)/do_k * ... * do_L/do_(L-1) * do_final/do_L * dloss/do_final) 
	for the current sample.
	The dimension is n_k X 1, or numOutputs X 1.
	Calculated from Model. Cached here. */
	Eigen::VectorXd currSample_chainRuleFactor; 
	/*For the k-th hidden layer, this stores do_k/do_(k-1).*/
	Eigen::MatrixXd currSample_dodinput;

	/*There are supposed to be n_k(numOutputs) elements in the vector.
	The jth element of the below vector is doutput/dw_j. The dimension for each
	element is p_k X n_k (numInputs X numOutputs). 
	Calculated from this Layer. Cached here. */
	std::vector<Eigen::MatrixXd> currSample_dodweights;
	/*The jth row of the matrix below is nothing but the transpose of the jth element of the
	currSample_dodweights times currSample_chainRuleFactor. The dimension is n_k X p_k.*/
	Eigen::MatrixXd currSample_neblaWeights;

	/*The vector contains n_k elements.
	The j-th element is nothing but doutput/dbias_j. The dimension is 1 X n_k.*/
	std::vector<Eigen::MatrixXd> currSample_dodbias;
	/*The jth row of the matrix below is nothing but the transpose of the jth element of
	currSample_dodbias times the currSample_chainRule. The dimension is n_k X 1.*/
	Eigen::MatrixXd currSample_neblaBias;

	/*Stores the sample-wise neblaWeight. The final nebla weights
	are nothing but the average of all elements in the vector. Each element has dimension
	n_k X p_k.*/
	std::vector<Eigen::MatrixXd> neblaWeights_bySampleVec;
	/*Stores sample-wise nebla bias. The final nebla bias are nothing but the average
	of all elements in this vector. Each element has dimension n_k X p_k.*/
	std::vector<Eigen::VectorXd> neblaBias_bySampleVec;

	/*Data-wise neblaWeights used in gradient-descent.*/
	Eigen::MatrixXd neblaWeights;
	/*Data-wise neblaBias used in gradient-descent.*/
	Eigen::VectorXd neblaBias;

	/*The activation function. It maps a scalar to another scalar. 
	Upon a hidden layer being added to the model, the user will be given the option 
	of passing in a user-defined activation. The setter of this field will be invoked
	to pass in that function. 
	The default actiavtion for a new hidden layer is the Bent Identity. See utilities.h.*/
	std::function<double(double)> activate_scalar;

	/*Maps the input to the output. Defined for convinient calculation of numerical diff.*/
	Eigen::VectorXd calcOutputFromInput(Eigen::VectorXd);

	/*/

	/*Maps the bias to the output. Use for convinient calculation of numerical diff.*/
	Eigen::VectorXd calcOutputFromBias(Eigen::VectorXd);

public:
	HiddenLayer(int numberInputs, int numberOutputs);

	void calcOutput() override;

	/*The broadcasted version for activation function.*/
	Eigen::VectorXd activate_vector(Eigen::VectorXd);

	Eigen::MatrixXd getWeights();
	void setWeights(Eigen::MatrixXd);
	
	/* Get the weights in the jth neuron, and return it as a column vector.
	Notice this is just getting the jth row of the weighting matrix, and
	returning its tranposed version. */
	Eigen::VectorXd getJthWeight(int j);


	Eigen::VectorXd getBias();
	void setBias(Eigen::VectorXd);

	Eigen::VectorXd getCurrSample_ChainRuleFactor();
	void setCurrSample_ChainRuleFactor(Eigen::VectorXd);

	Eigen::MatrixXd getCurrSample_dodinput();
	void setCurrSample_dodinput(Eigen::MatrixXd);

	/*Add all dodweights to the private vector currSample_dodweights. 
	Once again currSample_dodweights should have n_k elements, representing
	the derivative of output against the weights of n_k neurons, and 
	each element should have dimension p_k X n_k.*/
	void addCurrSample_dodweights();
	/*Returns the vector of matrices currSample_dodweights, who has n_k elements, representing
	the derivative of output against the weights of n_k neurons. The jth element
	is do_k/dw_kj. */
	std::vector<Eigen::MatrixXd> getCurrSample_dodweights();

	/*Add all dodbias to the private vector currSample_dodbias. 
	currSample_dodbias has n_k elements, with the jth representing 
	do_k/db_kj, the derivative of the output against the bias 
	of the jth neuron. Each element has dimension 1 X n_k. */
	void addCurrSample_dodbias();
	/*Returns the vector of matrices currSample_dodbias, who has n_k elements, representing
	the derivative of output against the bias of n_k neuron. The jth element
	is do_k/db_kj. */
	std::vector<Eigen::MatrixXd> getCurrSample_dodbias();

	Eigen::VectorXd getNeblaWeights();


	Eigen::VectorXd getNeblaBias();


	/*Lets the user pass in a self-defined activation function.*/
	void setActivation(double func(double));

};

