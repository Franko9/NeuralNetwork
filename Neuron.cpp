#include "Neuron.h"

Neuron::Neuron()
{
	this->value = 0.0;
	this->actVal = 0.0;
	this->diffVal = 0.0;
	this->activation = None;
}

Neuron::Neuron(double value, actFunc act)
{
	this->value = value;
	this->activation = act;
	this->activateValue();
}

void Neuron::setValue(double value)
{
	this->value = value;
}

double Neuron::getValue()
{
	return value;
}

void Neuron::activateValue()
{
	(this->*activationFunc[this->activation])();
}

void Neuron::differentiateOutput()
{
	(this->*differentiationFunc[this->activation])();
}

void Neuron::sigmoid()
{
	this->actVal = 1.0 / (1.0 + exp(-(this->value)));
}

void Neuron::reLU()
{
	if (this->value > 0.0)
	{
		this->actVal = value;
	}
	else
	{
		this->actVal = 0.0;
	}
}

void Neuron::noAct()
{
	this->actVal = this->value;
}

void Neuron::sigmoidPrime()
{
	this->diffVal = this->actVal * (1 - this->actVal);
}

void Neuron::reLUPrime()
{
	if (this->actVal > 0.0)
	{
		this->diffVal = 1.0;
	}
	else
	{
		this->diffVal = 0.0;
	}
}

void Neuron::leakyReLU()
{
	if (this->value > 0.0)
	{
		this->actVal = value;
	}
	else
	{
		this->actVal = 0.01*value;
	}
}

void Neuron::leakyReLUPrime()
{
	if (this->actVal > 0.0)
	{
		this->diffVal = 1.0;
	}
	else
	{
		this->diffVal = 0.01;
	}
}

void Neuron::noActPrime()
{
	this->diffVal = this->value;
}

double Neuron::getActivatedValue()
{
	return actVal;
}

void Neuron::printNeuron()
{
	cout << "Value: " << this->value << endl;
	cout << "Activation: " << this->actVal << endl;
}

double Neuron::getDifferentiatedValue()
{
	return this->diffVal;
}