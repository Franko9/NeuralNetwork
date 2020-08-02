#pragma once

#include <math.h>
#include <iostream>

using namespace std;

enum actFunc {
	None = 0,
	Sigmoid = 1,
	ReLU = 2,
	LeakyReLU = 3
};



class Neuron
{
public:
	Neuron();
	Neuron(double value = 0.0, actFunc act = Sigmoid);
	void setValue(double value);
	double getValue();
	void activateValue();
	void differentiateOutput();
	double getActivatedValue();
	double getDifferentiatedValue();
	void sigmoid();
	void sigmoidPrime();
	void reLU();
	void reLUPrime();
	void leakyReLU();
	void leakyReLUPrime();
	void noAct();
	void noActPrime();
	void printNeuron();
	
private:
	double value = 0.0;
	double actVal = 0.0;
	double diffVal = 0.0;
	actFunc activation;
	void (Neuron::* activationFunc[4])()		= { &Neuron::noAct,			&Neuron::sigmoid,		&Neuron::reLU,		&Neuron::leakyReLU };
	void (Neuron::* differentiationFunc[4])()	= { &Neuron::noActPrime,	&Neuron::sigmoidPrime,	&Neuron::reLUPrime, &Neuron::leakyReLUPrime };
};

