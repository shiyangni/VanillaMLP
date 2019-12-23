#include <iostream>
#include "Layer.h"
#include "Model.h"

using namespace std;

Layer::Layer(int numberInputs, int numberOutputs)
{
	numInputs = numberInputs;
	numOutputs = numberOutputs;
	layerIndex = -1;
 
}

/*Never invoked in practice. Used in testing.*/
Layer::Layer()
{
	numInputs = 0;
	numOutputs = 0;
	layerIndex = -1;
}

void Layer::readInput(Eigen::VectorXd input)
{
	inputVector = input;
}



void Layer::calcOutput()
{
	cout << "A General Layer is too abstract for calcOutput method to be implemented!" << endl;
}

void Layer::setLayerIndex(int i)
{
	layerIndex = i;
}

int Layer::getLayerIndex()
{
	return layerIndex;
}


