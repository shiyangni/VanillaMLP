#include <cstdlib>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <istream>
#include <vector>
#include <string>  

#include "Utilities.h"


using namespace std;
using namespace Eigen;

std::vector<std::vector<std::string>> csvToVecOfVecString(std::string fileName) {
	std::fstream file(fileName, std::ios::in);
	std::string csvLine;
	std::getline(file, csvLine); // Assuming the first line is the header.
	std::vector<std::vector<std::string>> result;
	while (std::getline(file, csvLine)) {
		std::istringstream csvLineStream(csvLine);
		vector<std::string> currRow;
		std::string currElement;
		while (std::getline(csvLineStream, currElement, ',')) {
			currRow.push_back(currElement);
		}
		result.push_back(currRow);
	}
	return result;
}

/*The function assumes our data only contains numerical inputs that can be coerced to double.*/
Eigen::MatrixXd readCSV(std::string fileName) {
	std::vector<std::vector<std::string>> temp = csvToVecOfVecString(fileName);
	int numRows = temp.size();
	int numColumns = temp.at(0).size();
	Eigen::MatrixXd result(numRows, numColumns);
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numColumns; j++) {
			result(i, j) = std::stod(temp.at(i).at(j));
		}
	}
	return result;
}

Eigen::MatrixXd numericDiff(Eigen::VectorXd input, Eigen::VectorXd func(VectorXd), double perturbance)
{
	VectorXd originalOutput = func(input);
	int outputDim = originalOutput.rows();
	int inputDim = input.rows();
	MatrixXd result = MatrixXd::Zero(inputDim, outputDim);
	for (int i = 0; i < inputDim; i++) {
		for (int j = 0; j < outputDim; j++) {
			VectorXd perturbedInput = input;
			perturbedInput(i) += perturbance;
			VectorXd perturbedOutput = func(perturbedInput);
			double original_yj = originalOutput(j);
			double perturbed_yj = perturbedOutput(j);
			result(i, j) = (perturbed_yj - original_yj) / perturbance;
		}
	}
	return result;
}

double identity(double x)
{
	return x;
}

double bentIdentity(double x)
{
	return (sqrt(pow(x,2) + 1) - 1) / 2 + x;
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}



