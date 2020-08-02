#pragma once

#include "Layer.h"
#include "Neuron.h"
#include "Matrix.h"

class NeuralNetwork
{
public:
	NeuralNetwork();
	NeuralNetwork(vector<int> topology, double out[], double err[], actFunc activation = ReLU);
	void setInput(vector<double> in);
	void setTarget(vector<double> t);
	void forward();
	void calculateError();
	void backprop();
	void updateWeights();
	void printNetwork();
	double getError();
	void setLearningRate(double lr);

private:
	vector<int> topology;
	Layer* input;
	int L;					// Number of layers in network (hidden and output)
	vector<Layer*> layers;	// Hidden and output layers
	vector<Matrix*> W;		// weights of layer
	vector<Matrix*> B;		// bias of layer
	vector<Matrix*> delta;	// delta values of layer (found durring backprop)
	vector<Matrix*> dW;		// weight update values (delta W)
	vector<Matrix*> dB;		// bias update values (delta B)
	Matrix* target;
	Matrix* errors;
	double E;
	double alpha = 1.0;
	double* err;
	double* out;
};

