#pragma once
#include <Eigen/Dense>

/* Layer is the abstract base class for the following three types of Layers:
- InputLayer
- OutputLayer
- HiddenLayer

All layers should be able to do the following:
- Initiate according to specfied number of inputs and outputs.
- Read in and store a given input, specified as an Eigen::VectorXd(column vector). When 
the layer exists within a model, it should be able to infer the source of input.
- Calculate the output and store it as a column vector.

We restrict all Layers to handle only vector inputs, and leave the stacking into matrix
to model. This is to ensure consistency with the vanilla NN structure.

Further we impose the restriction that a layer cannot back reference the containing
model. This is to simplify the pointer designs.
*/



class Layer
{
public:
	Layer(int numberInputs, int numberOutputs);
	/*The default constructor should never be called.*/
	Layer();
	/*The readInput with an vector argument simply reads in and stores the specified inputs. 
	Shared across all subclasses. But because a non-polymorphic overriden method is hidden from a sub-class,
	we make it virtual here. In all subclasses, implementation calls this function.*/
	void readInput(Eigen::VectorXd input);

	virtual void calcOutput();


	/*The getter and setter functions should be the same across all subclasses.*/
	int getNumInputs() { return numInputs; }
	void setNumInputs(int numberInputs) { numInputs = numberInputs; }
	
	int getNumOutputs() { return numOutputs; }
	void setNumOutputs(int numberOutputs) { numOutputs = numberOutputs; }
	
	Eigen::VectorXd getInput() { return inputVector; }
	void setInput(Eigen::VectorXd input) { inputVector = input; }
	/*Only to be invoked after input is read in, and 
	parameters are intialized. */
	Eigen::VectorXd getOutput() { return outputVector; }
	void setOutput(Eigen::VectorXd output) { outputVector = output; }
	




private:
	int numInputs;
	int numOutputs;
	Eigen::VectorXd inputVector;
	Eigen::VectorXd outputVector;

};






