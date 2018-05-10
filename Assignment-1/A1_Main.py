import A1_Defs

OriginalMatrix = A1_Defs.GetDataFromDataFile()
#A1_Defs.MatrixToString(OriginalMatrix) #this is just for testing

ColA, ColB, ColC = A1_Defs.GetThreeRandomNumbers(OriginalMatrix, -1, -1, -1)
Matrix1 = A1_Defs.MakeMatrix(OriginalMatrix, ColA, ColB, ColC, 'A')
#print(A1_Defs.MatrixToString(Matrix1)) #this is just for testing

ColD, ColE, ColF = A1_Defs.GetThreeRandomNumbers(OriginalMatrix, ColA, ColB, ColC)
Matrix2 = A1_Defs.MakeMatrix(OriginalMatrix, ColD, ColE, ColF, 'D')
#print(A1_Defs.MatrixToString(Matrix2)) #this is just for testing

Matrix3 = A1_Defs.AddingMatrices(Matrix1, Matrix2)
#print(A1_Defs.MatrixToString(Matrix3)) #this is just for testing

Matrix4 = A1_Defs.AddingContentOfEachRow(Matrix3)
#print(A1_Defs.MatrixToString(Matrix4)) #this is just for testing

Matrix5 = A1_Defs.SortElements(Matrix4, 'A')
#print(Matrix5) #this is just for testing

A1_Defs.PrintOutput(OriginalMatrix, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5)
