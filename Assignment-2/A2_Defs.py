import numpy as np
import sys
import os

class SquareMatrix:
    def __init__(self):
        self.n = 0

	# Requests that the user enter a dimension for the SquareMatrix
    def RequestDim(self):
        self.n = input("Please enter a positive number: ")

    # Reads in an input file and returns an n*n matrix composed of elements from that input file.
    def GetDataFromFile(self, filename):
        self.n = int(self.n)
        # If n is less than 3, return an error message.
        if self.n <= 3:
            print("Error: Input out of bounds.")
            sys.exit()
        # File contents are first copied to a Matrix.
        Matrix = np.loadtxt(filename, dtype=int)
        # Returns an error message if the Matrix's dimensions are less than n*n.
        if self.n*self.n > np.prod(Matrix.shape):
            print("Not enough elements in " + filename + ".")
            sys.exit()
        else:
            return Matrix[:self.n, :self.n]

    # Finds the product of two matrices.
    def MatrixProd(self, Matrix1, Matrix2):
        # Checks that Matrix1 has as many columns as Matrix2 has rows
        if Matrix1.shape[1] == Matrix2.shape[0]:
            return np.matmul(Matrix1, Matrix2)

    # Finds the dot product of two matrices
    def MatrixDotProd(self, Matrix1, Matrix2):
        # Checks that Matrix1 has as many columns as Matrix2 has rows
        if Matrix1.shape[1] == Matrix2.shape[0]:
            return np.dot(Matrix1, Matrix2)

    # Returns the transposed form of the original matrix.
    def Transpose(self, OrigMatrix):
        return np.transpose(OrigMatrix)

    # Divides one matrix by another.
    def DivideMatrix(self, Matrix1, Matrix2):
        Matrix3 = np.ndarray.astype(np.divide(Matrix1, Matrix2), dtype=object)
        Matrix3[Matrix3 == float('Inf')] = str("Undef")
        return Matrix3

    # Writes the parameter matrix to the output file.
    def OutputFile(self, Matrix = "No matrix", message = "No message"):
        with open(os.path.join(os.path.dirname(__file__), "Output.txt"), 'a') as fout:
            # print(str(message) + "\n" + str(Matrix) + "\n")
            fout.write(str(message) + "\n" + str(Matrix) + "\n")