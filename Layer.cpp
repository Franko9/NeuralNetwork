#include "Layer.h"

Layer::Layer()
{

}

Layer::Layer(int size, actFunc activation, int val, bool random)
{
	this->size = size;
	for (int i = 0; i < size; i++)
	{
		Neuron* n = new Neuron(val, activation);
		neurons.push_back(n);
	}
}

void Layer::printLayer()
{
	cout << endl << "----- Layer -----";
	cout << endl << "Net:\t";
	for (int i = 0; i < size; i++)
	{
		cout << neurons[i]->getValue() << "\t\t";
	}
	cout << endl << "Act:\t";
	for (int i = 0; i < size; i++)
	{
		cout << neurons[i]->getActivatedValue() << "\t\t";
	}
	cout << endl << "Dif:\t";
	for (int i = 0; i < size; i++)
	{
		cout << neurons[i]->getDifferentiatedValue() << "\t\t";
	}
	cout << endl;
}

void Layer::activateNeurons()
{
	for (int i = 0; i < size; i++)
	{
		neurons[i]->activateValue();
	}
}

void Layer::differentiateNeurons()
{
	for (int i = 0; i < size; i++)
	{
		neurons[i]->differentiateOutput();
	}
}

Matrix Layer::matrixifyOutput()
{
	Matrix result(this->size, 1);
	for (int i = 0; i < this->size; i++)
	{
		result.setValue(i, 0, this->neurons[i]->getActivatedValue());
	}
	return result;
}

Matrix Layer::matrixifyDiff()
{
	Matrix result(this->size, 1);
	for (int i = 0; i < this->size; i++)
	{
		result.setValue(i, 0, this->neurons[i]->getDifferentiatedValue());
	}
	return result;
}

void Layer::setValues(vector<double> values)
{
	for (int i = 0; i < size; i++)
	{
		neurons[i]->setValue(values[i]);
	}
}

void Layer::setValues(Matrix values)
{
	for (int i = 0; i < size; i++)
	{
		neurons[i]->setValue(values.at(i, 0));
	}
}

Neuron* Layer::getNeuron(int n)
{
	return this->neurons[n];
}