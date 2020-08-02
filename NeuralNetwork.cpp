#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::NeuralNetwork(vector<int> topology, double out[], double err[], actFunc activation)
{
	this->err = err;
	this->out = out;
	this->topology = topology;
	this->L = topology.size() - 1;
	this->input = new Layer(topology.front(), None, 1);
	this->target = new Matrix(topology.back(), 1);
	this->errors = new Matrix(topology.back(), 1);
	for (int l = 0; l < L; l++)
	{
		this->W.push_back(new Matrix(topology[l + 1], topology[l], true));
		this->dW.push_back(new Matrix(topology[l + 1], topology[l], true));
		this->delta.push_back(new Matrix(topology[l + 1], 1));
		this->B.push_back(new Matrix(topology[l + 1], 1));
		this->dB.push_back(new Matrix(topology[l + 1], 1));
		this->layers.push_back(new Layer(topology[l + 1], activation));
	}
}

void NeuralNetwork::setInput(vector<double> in)
{
	this->input->setValues(in);
}

void NeuralNetwork::setTarget(vector<double> t)
{
	for (int i = 0; i < this->topology.back(); i++)
	{
		this->target->setValue(i, 0, t[i]);
	}
}
void NeuralNetwork::calculateError()
{
	*this->errors = *target - layers.back()->matrixifyOutput();
	this->E = 0.0;
	for (int i = 0; i < topology.back(); i++)
	{
		this->E += this->errors->at(i, 0) * this->errors->at(i, 0);
	}
	
	for (int i = 0; i < topology[L]; i++)
	{
		err[i] = this->errors->at(i, 0);
	}
}

void NeuralNetwork::forward()
{
	input->activateNeurons();
	layers[0]->setValues(*W[0] * input->matrixifyOutput());
	layers[0]->activateNeurons();
	layers[0]->differentiateNeurons();

	for (int l = 1; l < L; l++)
	{
		layers[l]->setValues((*W[l] * layers[l - 1]->matrixifyOutput()) + *B[l]);
		layers[l]->activateNeurons();
		layers[l]->differentiateNeurons();
	}

	for (int i = 0; i < topology[L]; i++)
	{
		out[i] = layers[L - 1]->getNeuron(i)->getActivatedValue();
	}

	this->calculateError();
}

void NeuralNetwork::backprop()
{
	*delta[L-1] = errors->hadamardProduct(this->layers[L-1]->matrixifyDiff());
	*dW[L-1] = (delta[L-1]->outerProduct(this->layers[L - 2]->matrixifyOutput())).scalarMult(alpha);
	*dB[L-1] = delta[L-1]->scalarMult(alpha);

	for (int l = L - 2; l > 0; l--)
	{
		*delta[l] = (W[l + 1]->transpose() * *delta[l + 1]).hadamardProduct(this->layers[l]->matrixifyDiff());
		*dW[l] = (delta[l]->outerProduct(this->layers[l - 1]->matrixifyOutput())).scalarMult(alpha);
		*dB[l] = delta[l]->scalarMult(alpha);
	}

	*delta[0] = (W[1]->transpose() * *delta[1]).hadamardProduct(this->layers[0]->matrixifyDiff());
	*dW[0] = (delta[0]->outerProduct(this->input->matrixifyOutput())).scalarMult(alpha);
	*dB[0] = delta[0]->scalarMult(alpha);
}

void NeuralNetwork::updateWeights()
{
	for (int i = 0; i < L; i++)
	{
		*W[i] = *W[i] + *dW[i];
		*B[i] = *B[i] + *dB[i];
	}
}

void NeuralNetwork::printNetwork()
{
	input->printLayer();
	
	for (int l = 0; l < L; l++)
	{
		cout << "------- Hidden Layer " << l << " -----------" << endl;
		W[l]->printMatrix();
		B[l]->printMatrix();
		layers[l]->printLayer();
	}
}

double NeuralNetwork::getError()
{
	return this->E;
}

void NeuralNetwork::setLearningRate(double lr)
{
	this->alpha = lr;
}