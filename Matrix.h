#pragma once
#include <iostream>
#include <vector>
#include <time.h>

using namespace std;

class Matrix
{
public:
	Matrix();
	Matrix(int rows, int cols, bool random = false, double value = 0.0);
	void setValue(int row, int col, double value);
	double at(int row, int col);
	void printMatrix();
	Matrix transpose(); // Returns transposed matrix
	Matrix multiply(Matrix multiplicand); // Returns product of 2 matrices
	Matrix scalarMult(double multiplicand); // Returns scaled matrix
	Matrix hadamardProduct(Matrix multiplicand); // Returns elemental product of matrices
	Matrix outerProduct(Matrix multiplicand); // Returns matrix of outer product of 2 vectors (n x 1 matrices)
	Matrix operator*(const Matrix& multiplicand);
	Matrix operator+(const Matrix& addend);
	Matrix operator-(const Matrix& sub);
	void operator=(const Matrix& result);

private:
	int rows;
	int cols;
	vector<vector<double>> values;

};

