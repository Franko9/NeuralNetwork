#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Neuron.h"
#include "Layer.h"
#include <vector>
#include <fstream>

using namespace std;

int main()
{
	vector<int> top{2, 7, 1, 1};
	double output[1];
	double error[1];

	NeuralNetwork nn(top, output, error, LeakyReLU);
	nn.setLearningRate(0.1);

	vector<double> in1{ 0.01, 0.01 };
	vector<double> in2{ 0.01, 0.99 };
	vector<double> in3{ 0.99, 0.01 };
	vector<double> in4{ 0.99, 0.99 };

	vector<double> targ1{ 0.01 };
	vector<double> targ2{ 0.99 };
	vector<double> targ3{ 0.99 };
	vector<double> targ4{ 0.01 };

	ofstream outputFile;
	outputFile.open("E:/NeuralNetworkOutput/output.csv");
	outputFile << "Error1, Input1, Input2, Output, Target, Error2, Input1, Input2, Output, Target, Error3, Input1, Input2, Output, Target, Error4, Input1, Input2, Output, Target, Total Error\n";

	for (int i = 0; i < 10000; i++)
	{
		cout << i << endl;
		nn.setInput(in1);
		nn.setTarget(targ1);
		nn.forward();
		nn.backprop();
		nn.updateWeights();
		outputFile << error[0] << ", " << in1[0] << ", " << in1[1] << ", " << output[0] << ", " << targ1[0] << ", ";

		nn.setInput(in2);
		nn.setTarget(targ2);
		nn.forward();
		nn.backprop();
		nn.updateWeights();
		outputFile << error[0] << ", " << in2[0] << ", " << in2[1] << ", " << output[0] << ", " << targ2[0] << ", ";

		nn.setInput(in3);
		nn.setTarget(targ3);
		nn.forward();
		nn.backprop();
		nn.updateWeights();
		outputFile << error[0] << ", " << in3[0] << ", " << in3[1] << ", " << output[0] << ", " << targ3[0] << ", ";

		nn.setInput(in4);
		nn.setTarget(targ4);
		nn.forward();
		nn.backprop();
		nn.updateWeights();
		outputFile << error[0] << ", " << in4[0] << ", " << in4[1] << ", " << output[0] << ", " << targ4[0] << ", " << nn.getError() << "\n";
	}

	nn.setInput(in1);
	nn.setTarget(targ1);
	nn.forward();
	cout << endl << endl << "Input 0 0" << endl;
	nn.printNetwork();


	nn.setInput(in2);
	nn.setTarget(targ2);
	nn.forward();
	cout << endl << endl << "Input 0 1" << endl;
	nn.printNetwork();

	nn.setInput(in3);
	nn.setTarget(targ3);
	nn.forward();
	cout << endl << endl << "Input 1 0" << endl;
	nn.printNetwork();

	nn.setInput(in4);
	nn.setTarget(targ4);
	nn.forward();
	cout << endl << endl << "Input 1 1" << endl;
	nn.printNetwork();
	
	outputFile.close();

	return 0;
}