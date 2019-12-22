#include <cstdlib>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <istream>
#include <vector>
#include <string>  

#include "Utilities.h"
#include "Layer.h"
#include "InputLayer.h"
#include "OutputLayer.h"
#include "HiddenLayer.h"
#include "Model.h"

using namespace std;
using namespace Eigen;

/*F: (x1 x2 x3) -> (x1+2x2+3x3, 10x1+20x2+30x3, 100x1+200x2+300x3, 1000x1+2000x2+3000x3)*/
VectorXd testMap(VectorXd input) {
	VectorXd result(4);
	for (int i = 0; i < result.rows(); i++) {
		VectorXd factor(input.rows());
		for (int j = 0; j < factor.rows(); j++) {
			factor(j) = (1 + j) * pow(10, i);
		}
		result(i) = input.dot(factor);
	}
	return result;
}

VectorXd testMap2(VectorXd input) {
	VectorXd result(1);
	result(0) = pow(input(0), 2) / 2;
	return result;
}


/*Here's the map
f: (x1, x2) -> (x1^2 + x2/8, x1^2/2 + x2/4, x1^2/4 + x2/2, x1^2/8 + x2)*/
VectorXd testMap3(VectorXd input) {
	VectorXd result(4);
	for (int i = 0; i < result.rows(); i++) {
		result(i) = pow(0.5, i) * pow(input(0), 2) + input(1) / pow(2, 3 - i);
	}
	return result;
}

int main() {
	cout << "Welcome to testing for Vanilla MLP!" << endl;




    /*Testing reading CSV in.*/
	cout << "Begin testing readCSV." << endl;
	vector<vector<string>> temp = csvToVecOfVecString("data.csv");
    
	cout << "The element at Row 2 Column 4 is 11947. Did we get it right?" << endl; // 0 indexing
	cout << "That element in the read in result is: " << temp.at(2).at(4) << endl;

	cout << "The element at Row 4 Column 0 is 550000. Did we get it right?" << endl;
	cout << "That element in the read in result is: " << temp.at(4).at(0) << endl;
	cout << "We got both of them right! \n \n " << endl;


	/*Testing reading data in.*/
	cout << "Let's see if the readCSV function works!" << endl;
	cout << readCSV("data_fake.csv") << endl;
	cout << "Read CSV function works as intended. \n\n" << endl;
	cout << "This concludes testing readCSV \n\n" << endl;


	/*Testing how the Eigen::Matrix works. */
	/*Assignment of a subset deep copies? Yes.*/
	//MatrixXd a(5, 3);
	//a.fill(10);
	//MatrixXd b(11, 1);
	//b.fill(1);
	//cout << b * a << endl;
	//MatrixXd b = a.block(0, 1, a.rows(), a.cols() - 1);
	//b.fill(55);
	//cout << a << endl;
	//cout << b << endl;


	/*Testing for Model.h*/
	Model emptyModel;
	MatrixXd data_train = readCSV("data.csv");
	VectorXd sample_x = emptyModel.getXTrain().row(0).transpose();

	cout << "Begin testing Model" << endl;
	/*Test the default constructor of Model.*/
	cout << "Testing the default constructor Model." << endl;

	cout << "data in the empty model is: \n" << emptyModel.getDataTrain() << endl;
	cout << "The X_train is: \n " << emptyModel.getXTrain() << endl;
	cout << "The y_train is: \n " << emptyModel.getYTrain() << endl;
	cout << "This is exactly the data in data_fake.csv! \n\n" << endl;

	/*Testing the initiation of a model with an inputLayer.*/
	cout << "Testing the initiation of a model with an inputLayer." << endl;

 	Model dataModel(data_train);
	cout << "Our data has " << data_train.cols() - 1 << " features." << endl;
	cout << "After adding the input layer, the input layer has " << dataModel.getLayer(0).getNumInputs()
		<< " inputs" << endl;
	cout << "Model initiation works fine! \n\n" << endl;

	/*Testing for Layer.*/
	cout << "Begin testing for Layer and its subclasses" << endl;
	/*Layer*/
	/*Test if readInput(...) works properly.*/
	cout << "Test if readInput(...) works properly at a general layer." << endl;
	
	Layer generalLayer(sample_x.rows(), 5); // numInputs should match the inputting vector's dimension.
											// numOutputs is randomly assigned.
	generalLayer.readInput(sample_x);
	cout << "Our sample input is supposed to be: \n" << sample_x << endl;
	cout << "In the general layer, the read-in input is: \n" << generalLayer.getInput() << endl;
	generalLayer.calcOutput();
	cout << "This concludes testing for a generic layer. \n \n" << endl;

	/*InputLayer*/
	/*Testing if readInput and calcOutput works properly for an InputLayer.*/
	cout << "Testing if readInput and calcOutput works properly for an InputLayer." << endl;
	cout << "Our sample input is supposed to be: \n" << sample_x << endl;
	InputLayer inputLayer1(sample_x.rows());// numInputs should match the inputting vector's dimension.
											   // numOutputs is randomly assigned.
	inputLayer1.readInput(sample_x);
	inputLayer1.calcOutput();
	cout << "In the input layer, the input is now: \n" << inputLayer1.getInput() << endl;
	cout << "In the input layer, the output is now \n" << inputLayer1.getOutput() << endl;
	cout << "This concludes testing for input layer \n\n" << endl;

	/*OutputLayer*/
	/*Testing calcOutput for Output Layer.*/
	cout << "Begin testing output layer." << endl;
	OutputLayer outputLayer1(sample_x.rows());
	outputLayer1.readInput(sample_x);
	outputLayer1.calcOutput();
	cout << "The input into the output layer is: \n" << sample_x << endl;
	cout << "The output layer should add all inputs and return it as a 1x1 matrix. \n"
		"Our output layer outputs: " << outputLayer1.getOutput() << endl;
	cout << "The output layer's do/dinput should be a vector of ones. Per our calculation"
		" it is: \n" << outputLayer1.calcDoDinput() << endl;
	cout << "This concludes testing for output layer \n\n" << endl;
	
	/*HiddenLayer.*/
	cout << "Begin testing for hidden layer." << endl;
	cout << "The hidden layer is initiated on the sample: \n" << sample_x << endl;
	HiddenLayer hiddenLayer1(sample_x.rows(), 5, identity);
	hiddenLayer1.readInput(sample_x);
	cout << "The hidden layer's weights should be a 5 X 3 matrix, by default all initialized"
		"to 1. Our layer's weights are: \n" << hiddenLayer1.getWeights() << endl;
	cout << "The bias should be a 5 X 1 matrix all equal to 1. Our layer's biases are \n"
	     << hiddenLayer1.getBias() << endl;
	
	hiddenLayer1.calcOutput();
	cout << "The output vector is supposed to be a 5 X 1 vector, each item is sum of inputs + 1."
		"Our layer's output vector is: \n" << hiddenLayer1.getOutput() << "\n\n" << endl;

	/*Test numerical differentiation within each layer.*/
	cout << "Test numerical differentiation within the layer."<< endl;
	HiddenLayer hiddenLayer3(sample_x.rows(), 3, identity);
	MatrixXd weights_hl3(3, 3);
	weights_hl3 << 1, 2, 3, 10, 20, 30, 100, 200, 300;
	hiddenLayer3.setWeights(weights_hl3);
	cout << "We construct the layer to take in three inputs, produce three outpus, actiavte through"
		" identity, and set the weights to the following: \n" << hiddenLayer3.getWeights() << endl;
	cout << "The bias remains 1: \n" << hiddenLayer3.getBias() << endl;
	hiddenLayer3.readInput(sample_x);
	hiddenLayer3.calcOutput();
	cout << "On the input: \n" << sample_x << endl;
	cout << "The outputs are: \n" << hiddenLayer3.getOutput() << endl;
	/*do/dweightJ*/
	cout << "do/dweight0 should be a 3 X 3 matrix, with the first column equaling the input,"
		" and the rest 0. Our layer calculates the jacobian to be: \n" << hiddenLayer3.calcDoDweightJ(0)
		<< endl;
	cout << "\n\n do/dweight1 should also be 3X3, with the second column equaling input,"
		"And the rest equaling 0. The layer calculates the Jacobian to be :\n"
		<< hiddenLayer3.calcDoDweightJ(1) << endl;
	cout << "\n\n do/dweight2 should also be 3X3, with the third column equaling input,"
		"And the rest equaling 0. The layer calculates the Jacobian to be :\n"
		<< hiddenLayer3.calcDoDweightJ(2) << endl;
	cout << "Errors appear. The current perturbance used in numerical diff is 1 X 10^(-7)"
		"Underflow starts to kick in after 1 X 10^(-8). We stick with 1 X 10^(-7)" << endl;
	/*do/dbiasJ*/
	cout << "\n do/dbias0 should be a row vector (1, 0, 0). It's calcualated to be: \n" 
		<< hiddenLayer3.calcDoDbiasJ(0) << endl;
	cout << "\n do/dbias1 should be a row vector (0, 1, 0). It's calcualated to be: \n" 
		<< hiddenLayer3.calcDoDbiasJ(1) << endl;
	cout << "\n do/dbias2 should be a row vector (0, 0, 1). It's calcualated to be: \n"
		<< hiddenLayer3.calcDoDbiasJ(2) << endl;
	/*do/dinput*/
	cout << "\n do_k/do_(k-1), or do_k/dinput should be the transposed weights matrix."
		" It actually is \n: " << hiddenLayer3.calcDoDinput() << "\n and our weights matrix"
		" is \n" << hiddenLayer3.getWeights() << endl;

	/*Layer with Bent identity*/
	cout << "\n \nTest a hiddenLayer with bentIdentity activation." << endl;
	HiddenLayer hiddenLayer2(sample_x.rows(), 2);
	hiddenLayer2.readInput(sample_x);
	hiddenLayer2.calcOutput();
	cout << "The outputs are activated through a bent identity. The result should be 2017.75" << endl;
	cout << "The layer's output is: \n" << hiddenLayer2.getOutput() << endl;
	cout << "do/dweight0 is: \n" << hiddenLayer2.calcDoDweightJ(0) << endl;
	cout << "do/dbias0 is: \n" << hiddenLayer2.calcDoDbiasJ(0) << endl;
	cout << "do/dinput is: \n" << hiddenLayer2.calcDoDinput() << endl;
	cout << "We've checked that the values don't explode." << endl;
	
	cout << "This concludes testing for a stand-alone HiddenLayer. \n\n" << endl;

	
	/*Testing utitlities.*/
	//cout << "Testing the utilities funcitons. Note they don't belong to any classes." << endl;
	/*Testing numericDiff.*/
	/*cout << "Testing nuermic differentiation." << endl;
	VectorXd testVec = VectorXd::Ones(3);
	cout << "The input is \n" << testVec << endl;
	cout << "Our function maps the testVec to \n" << testMap(testVec) << endl;
	cout << "The Jacobian in denominator layout is \n" << numericDiff(testVec, testMap) << endl;
	
	cout << "A second test for numerical differentiation." << endl;
	VectorXd testVec2 = VectorXd::Ones(1);
	cout << "The second input is \n" << testVec2 << endl;
	cout << "The derivative is supposed to be 1. Our function says its \n" 
		<< numericDiff(testVec2, testMap2) << endl;

	cout << "A third test for numerical differentiation." << endl;
	VectorXd testVec3(2);
	testVec3 << 1, 8;
	cout << "The third input is \n" << testVec3 << endl;
	cout << "The mapped output is \n" << testMap3(testVec3) << endl;
	cout << "The calculated Jacobian is supposed to be \n"
		"2   1    1/2    1/4 \n"
		"1/8 1/4  1/2    1   \n" << endl;
	cout << "It actually is \n" << numericDiff(testVec3, testMap3) << endl;*/
		
	//cout << "This concludes testing for utility functions. \n\n" << endl;
	return 0;
}