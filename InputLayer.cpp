#include "InputLayer.h"
#include <iostream>

using namespace std;

InputLayer::InputLayer(int numberInputs)
{
	Layer(numberInputs, numberInputs);
	setNumInputs(numberInputs);
	setNumOutputs(numberInputs);
}


/*Invoked only after input has been read in.*/
void InputLayer::calcOutput()
{
	setOutput(getInput());
}


