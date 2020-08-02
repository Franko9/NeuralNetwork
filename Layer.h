#pragma once

#include "Neuron.h"
#include <vector>
#include "Matrix.h"

using namespace std;

class Layer
{
public:
	Layer();
	Layer(int size, actFunc activation = ReLU, int val = 0, bool random = false);
	void activateNeurons();
	void differentiateNeurons();
	Matrix matrixifyOutput();
	Matrix matrixifyDiff();
	void setValues(vector<double> values);
	void setValues(Matrix values);
	void printLayer();
	Neuron* getNeuron(int n);

private:
	int size = 0;
	vector<Neuron*> neurons;
};

