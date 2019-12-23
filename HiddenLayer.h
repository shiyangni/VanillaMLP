#pragma once
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include "Layer.h"
#include "Utilities.h"

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
	/*For the k-th hidden layer, this stores do_k/do_(k-1). Of dimension p_k X n_k.*/
	Eigen::MatrixXd currSample_DoDinput;

	/*There are supposed to be n_k(numOutputs) elements in the vector.
	The jth element of the below vector is doutput/dw_j. The dimension for each
	element is p_k X n_k (numInputs X numOutputs). 
	Calculated from this Layer. Cached here. */
	std::vector<Eigen::MatrixXd> currSample_DoDweights;
	/*The jth row of the matrix below is nothing but the transpose of the jth element of the
	currSample_DoDweights times currSample_chainRuleFactor. The dimension is n_k X p_k.*/
	Eigen::MatrixXd currSample_neblaWeights;

	/*The vector contains n_k elements.
	The j-th element is nothing but doutput/dbias_j. The dimension is 1 X n_k.*/
	std::vector<Eigen::MatrixXd> currSample_DoDbias;
	/*The jth row of the matrix below is nothing but the transpose of the jth element of
	currSample_DoDbias times the currSample_chainRule. The dimension is n_k X 1.*/
	Eigen::MatrixXd currSample_neblaBias;

	/*Stores the sample-wise neblaWeight. The final nebla weights
	are nothing but the average of all elements in the vector. Each element has dimension
	n_k X p_k.*/
	std::vector<Eigen::MatrixXd> neblaWeights_bySample;
	/*Stores sample-wise nebla bias. The final nebla bias are nothing but the average
	of all elements in this vector. Each element has dimension n_k X p_k.*/
	std::vector<Eigen::VectorXd> neblaBias_bySample;

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

	/*Returns the output based on current configuration.*/
	Eigen::VectorXd returnOutput();


public:
	/*Never invoked.*/
	HiddenLayer();

	/*By default all weights and biases are initialized to one. 
	The activation function is Bent identity by default.*/ 
	HiddenLayer(int numberInputs, int numberOutputs, std::function<double(double)> = bentIdentity);

	void calcOutput() override;

	/*The broadcasted version for activation function.*/
	Eigen::VectorXd activate_vector(Eigen::VectorXd);

	Eigen::MatrixXd getWeights();
	void setWeights(Eigen::MatrixXd);
	

	Eigen::VectorXd getBias();
	void setBias(Eigen::VectorXd);

	Eigen::VectorXd getCurrSample_ChainRuleFactor();
	void setCurrSample_ChainRuleFactor(Eigen::VectorXd);

	/* Get the weights in the jth neuron, and return it as a column vector.
	Notice this is just getting the jth row of the weighting matrix, and
	returning its tranposed version. */
	Eigen::VectorXd getJthWeight(int j);

	/*Calculates do_k/do_(k-1). 
	The implementation performs nuermical differentiation
	in the context, i.e., doesn't invoke the numericDiff in utitlies.h. This
	breaks the abstraction barrier between Layer and Numeric Methods. Hope to improve on
	this in future iterations.*/
	Eigen::MatrixXd calcDoDinput(double perturbance = 0.000001);

	/*Returns the output's Jacobian againt input at current values of input, 
	weights, and bias.*/
	Eigen::MatrixXd getCurrSample_DoDinput();

	/*Calculates the do_k/dw_kj. 
	The implementation performs nuermical differentiation
	in the context, i.e., doesn't invoke the numericDiff in utitlies.h. This
	breaks the abstraction barrier between Layer and Numeric Methods. Hope to improve on
	this in future iterations. */
	Eigen::MatrixXd calcDoDweightJ(int j, double perturbance = 0.000001);
	/*Add all DoDweights to the private vector currSample_DoDweights. 
	Once again currSample_DoDweights should have n_k elements, representing
	the derivative of output against the weights of n_k neurons, and 
	each element should have dimension p_k X n_k.*/
	void calcCurrSample_DoDweights();
	/*Returns the vector of matrices currSample_DoDweights, who has n_k elements, representing
	the derivative of output against the weights of n_k neurons. The jth element
	is do_k/dw_kj. */
	std::vector<Eigen::MatrixXd>& getCurrSample_DoDweights();


	/*Calculates do_k/db_k. The implementation performs nuermical differentiation
	in the context, i.e., doesn't invoke the numericDiff in utitlies.h. This
	breaks the abstraction barrier between Layer and Numeric Methods. Hope to improve on
	this in future iterations.*/
	Eigen::MatrixXd calcDoDbiasJ(int j, double perturbance = 0.000001);
	/*Add all DoDbias to the private vector currSample_DoDbias. 
	currSample_DoDbias has n_k elements, with the jth representing 
	do_k/db_kj, the derivative of the output against the bias 
	of the jth neuron. Each element has dimension 1 X n_k. */
	void calcCurrSample_DoDbias();
	/*Returns the vector of matrices currSample_DoDbias, who has n_k elements, representing
	the derivative of output against the bias of n_k neuron. The jth element
	is do_k/db_kj. */
	std::vector<Eigen::MatrixXd>& getCurrSample_DoDbias();

    /*Calculate all intermediate products.*/
	void calcJacobians();

	Eigen::VectorXd getNeblaWeights();

	Eigen::VectorXd getNeblaBias();

	/*Lets the user pass in a self-defined activation function.*/
	void setActivation(std::function<double(double)> func);

};

