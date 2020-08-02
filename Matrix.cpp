#include "Matrix.h"

Matrix::Matrix()
{
	rows = 0;
	cols = 0;
}

Matrix::Matrix(int rows, int cols, bool random, double value)
{
	this->rows = rows;
	this->cols = cols;

	if (!random)
	{
		for (int r = 0; r < rows; r++)
		{
			vector<double> row;
			for (int c = 0; c < cols; c++)
			{
				row.push_back(value);
			}
			values.push_back(row);
		}
	}
	else {
		srand(time(0));
		for (int r = 0; r < rows; r++)
		{
			vector<double> row;
			for (int c = 0; c < cols; c++)
			{
				row.push_back(rand()/10000.00);
			}
			values.push_back(row);
		}
	}
}

Matrix Matrix::transpose()
{
	Matrix result(cols, rows);
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			result.values[c][r] = this->values[r][c];
		}
	}
	return result;
}

void Matrix::setValue(int row, int col, double value)
{
	values[row][col] = value;
}

double Matrix::at(int row, int col)
{
	return this->values[row][col];
}

Matrix Matrix::multiply(Matrix multiplicand)
{
	Matrix product(rows, multiplicand.cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < multiplicand.cols; j++)
		{
			for (int k = 0; k < cols; k++)
			{
				product.values[i][j] += this->values[i][k] * multiplicand.values[k][j];
			}
		}
	}
	return product;
}

void Matrix::printMatrix()
{
	for (int r = 0; r < rows; r++)
	{
		cout << "[ ";
		for (int c = 0; c < cols; c++)
		{
			cout << this->values[r][c] << "\t\t";
		}
		cout << "]" << endl;
	}
}

Matrix Matrix::operator*(const Matrix& multiplicand)
{
	return this->multiply(multiplicand);
}

void Matrix::operator=(const Matrix& result)
{
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			this->values[r][c] = result.values[r][c];
		}
	}
}

Matrix Matrix::scalarMult(double multiplicand)
{
	Matrix product(rows, cols);
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			product.values[r][c] = this->values[r][c] * multiplicand;
		}
	}
	return product;
}

Matrix Matrix::operator+(const Matrix& addend)
{
	Matrix result(rows, cols);
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			result.values[r][c] = this->values[r][c] + addend.values[r][c];
		}
	}
	return result;
}

Matrix Matrix::operator-(const Matrix& sub)
{
	Matrix result(rows, cols);
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			result.values[r][c] = this->values[r][c] - sub.values[r][c];
		}
	}
	return result;
}

Matrix Matrix::hadamardProduct(Matrix multiplicand)
{
	Matrix product(rows, cols);
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			product.values[r][c] = this->values[r][c] * multiplicand.values[r][c];
		}
	}
	return product;
}

Matrix Matrix::outerProduct(Matrix multiplicand)
{
	Matrix product(this->rows, multiplicand.rows);

	for (int i = 0; i < this->rows; i++)
	{
		for (int j = 0; j < multiplicand.rows; j++)
		{
			product.values[i][j] = this->values[i][0] * multiplicand.values[j][0];
		}
	}
	return product;
}