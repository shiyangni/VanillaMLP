#include <iostream>
#include "Layer.h"
#include "Model.h"

using namespace std;

Layer::Layer(int numberInputs, int numberOutputs)
{
	numInputs = numberInputs;
	numOutputs = numberOutputs;

 
}

/*Never invoked in practice. Used in testing.*/
Layer::Layer()
{
	numInputs = 0;
	numOutputs = 0;

}

void Layer::readInput(Eigen::VectorXd input)
{
	inputVector = input;
}



void Layer::calcOutput()
{
	cout << "A General Layer is too abstract for calcOutput method to be implemented!" << endl;
}


