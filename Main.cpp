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

	///*Getting familiar with matrix initialization and arithmetics.*/
	//MatrixXd m(5, 2);
	//m.fill(0);
	//cout << "First matrix is: \n" << m << endl;

	//VectorXd c(5);
	//c.fill(1);
	//cout << "First columns vector is: \n" << c << endl;

	//RowVectorXd r(10);
	//r.fill(3);
	//cout << "First row vector is: \n" << r << endl;


    /*Testing reading CSV in.*/
	cout << "Testing the utility function that reads csv into a vector of string vectors." << endl;
	vector<vector<string>> temp = csvToVecOfVecString("data.csv");
    
	cout << "The element at Row 2 Column 4 is 11947. Did we get it right?" << endl; // 0 indexing
	cout << "That element in the read in result is: " << temp.at(2).at(4) << endl;

	cout << "The element at Row 4 Column 0 is 550000. Did we get it right?" << endl;
	cout << "That element in the read in result is: " << temp.at(4).at(0) << endl;
	cout << "We got both of them right! \n \n \n" << endl;


	/*Testing reading data in.*/
	cout << "Let's see if the readCSV function works!" << endl;
	cout << readCSV("data_fake.csv") << endl;
	cout << "Read CSV function works as intended. \n\n" << endl;

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

	//cout << "Begin testing Model" << endl;
	///*Test the default constructor of Model.*/
	//cout << "Testing the default constructor Model." << endl;

	//cout << "data in the empty model is: \n" << emptyModel.getDataTrain() << endl;
	//cout << "The X_train is: \n " << emptyModel.getXTrain() << endl;
	//cout << "The y_train is: \n " << emptyModel.getYTrain() << endl;
	//cout << "This is exactly the data in data_fake.csv! \n\n" << endl;

	///*Testing the initiation of a model with an inputLayer.*/
	//cout << "Testing the initiation of a model with an inputLayer." << endl;

 //	Model dataModel(data_train);
	//cout << "Our data has " << data_train.cols() - 1 << " features." << endl;
	//cout << "After adding the input layer, the input layer has " << dataModel.getLayer(0).getNumInputs()
	//	<< " inputs" << endl;
	//cout << "Model initiation works fine! \n\n" << endl;

	/*Testing for Layer.h and subclasses.*/
	cout << "Testing for Layer and its subclasses" << endl;
	/*Layer*/
	/*Test if readInput(...) works properly.*/
	cout << "Test if readInput(...) works properly at a general layer." << endl;
	VectorXd sample_x = emptyModel.getXTrain().row(0).transpose();
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
	cout << "Testing calcOutput for Output Layer." << endl;
	OutputLayer outputLayer1(sample_x.rows());
	outputLayer1.readInput(sample_x);
	outputLayer1.calcOutput();
	cout << "The output layer should add all inputs and return it as a 1x1 matrix. \n"
		"Our output layer outputs: " << outputLayer1.getOutput() << endl;
	cout << "This concludes testing for output layer \n\n" << endl;
	
	
	
	/*Testing utitlities.*/
	cout << "Testing the utilities funcitons. Note they don't belong to any classes." << endl;
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
		
	cout << "This concludes testing for utility functions. \n\n" << endl;
	return 0;
}