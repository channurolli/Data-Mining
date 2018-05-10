import time                 #provides timing for benchmarks
from numpy  import *        #provides complex math and array functions
from sklearn import svm     #provides Support Vector Regression
import csv

#Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class FitnessAnalyzer:
    def __init__(self):
        self.filedata = FromDataFileMLR.DataFromFile()
        self.fitnessdata = FromFinessFileMLR.FitnessResults()
    #------------------------------------------------------------------------------
    def getAValidrow(self, numOfFea, eps=0.015):
        sum = 0
        while (sum < 3):
           V = zeros(numOfFea)
           for j in range(numOfFea):
              r = random.uniform(0,1)
              if (r < eps):
                 V[j] = 1
              else:
                 V[j] = 0
           sum = V.sum()
        return V
    #------------------------------------------------------------------------------
    def Create_A_Population(self, numOfPop, numOfFea):
        population = random.random((numOfPop,numOfFea))
        for i in range(numOfPop):
            V = self.getAValidrow(numOfFea)
            for j in range(numOfFea):
                population[i][j] = V[j]
        return population
    #------------------------------------------------------------------------------
    # The following creates an output file. Every time a model is created the
    # descriptors of the model, the ame of the model (ex: "MLR" for multiple
    # linear regression of "SVM" support vector machine) the R^2 of training, Q^2
    # of training,R^2 of validation, and R^2 of test is placed in the output file
    def createAnOutputFile(self):
        file_name = None
        algorithm = None

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if ( (file_name == None) and (algorithm != None)):
            file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__,
                            algorithm.model.__class__.__name__, algorithm.gen_max,timestamp)
        elif file_name==None:
            file_name = "{}.csv".format(timestamp)
        fileOut = open(file_name, 'w')
        fileW = csv.writer(fileOut)

        fileW.writerow(['Descriptor ID', 'Fitness', 'Model','R2', 'Q2', \
                'R2Pred_Validation', 'R2Pred_Test'])

        return fileW
    #-------------------------------------------------------------------------------------------
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
    #   NewPopulation = create a 2D array of (numOfPop by num of features)
    #   sort the OldPopulation and their fitness value based on the asending
    #   order of the fitness. The lower is the fitness, the better it is.
    #   So, Move two rows with of the OldPopulation with the lowest fitness
    #   to row 1 and row 2 of the new population.
        NewPopulation = ndarray((numOfPop, numOfFea))
        temp = ndarray(numOfFea)
        F = 0.5
        CV = 0.7 #crossover value
        RowT = 0
        RowR = 0
        RowS = 0

        # Sort OldPopulation from best fitness at position 0 to worst at position 1
        for r in range(0, numOfPop):
            BestFitInd = r
            for r2 in range(r, numOfPop):
                if (fitness[r] < fitness[BestFitInd]) & (fitness[r] > 0):
                    BestFitInd = r
            copyto(temp, OldPopulation[r])
            copyto(OldPopulation[r], OldPopulation[BestFitInd])
            copyto(OldPopulation[r], temp)
        copyto(NewPopulation[0], OldPopulation[0])

        for row in range(1, numOfPop):
            # Ensuring that values of RowT, RowR, and RowS are all random and distinct
            while True:
                RowT = random.randint(1, numOfPop)
                if RowT != row:
                    break
            while True:
                RowR = random.randint(1, numOfPop)
                if RowR != row & RowR != RowT:
                    break
            while True:
                RowS = random.randint(1, numOfPop)
                if RowS != row & RowS != RowR & RowS != RowT:
                    break

            # Nested for loop calculates the value of each element in NewPopulation
            V = ndarray(numOfFea)
            for col in range(0, numOfFea):
                V[col] = OldPopulation[RowT][col] + (F * (OldPopulation[RowR][col] - OldPopulation[RowS][col]))
                if random.rand(0,1) > CV:
                    V[col] = OldPopulation[row][col]
            copyto(NewPopulation[row], V)

        return NewPopulation
    #-------------------------------------------------------------------------------------------
    def PerformOneMillionIteration(self, numOfPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 1
        OldPopulation = population
        while (NumOfGenerations < 1000):
            population = self.createANewPopulation(numOfPop, numOfFea, OldPopulation, fitness)
            fittingStatus, fitness = self.fitnessdata.validate_model(model,fileW, population, \
                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
            NumOfGenerations = NumOfGenerations + 1
            print(NumOfGenerations)
        return

#--------------------------------------------------------------------------------------------
def main():
    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()
    filedata = FromDataFileMLR.DataFromFile()
    fitnessdata = FromFinessFileMLR.FitnessResults()
    analyzer = FitnessAnalyzer()

    # create an output file. Name the object to be FileW 
    fileW = analyzer.createAnOutputFile()

    #Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50 
    numOfFea = 385

    # we continue exhancing the model; however if after 1000 iteration no
    # enhancement is done, we can quit
    unfit = 1000

    # Final model requirements: The following is used to evaluate each model. The minimum
    # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
    # test should be 0.5
    R2req_train    = .6 
    R2req_validate = .5
    R2req_test     = .5

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = filedata.getAllOfTheData()
    TrainX, ValidateX, TestX = filedata.rescaleTheData(TrainX, ValidateX, TestX)

    fittingStatus = unfit
    population = analyzer.Create_A_Population(numOfPop,numOfFea)
    fittingStatus, fitness = fitnessdata.validate_model(model,fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    analyzer.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW, \
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#main routine ends in here
#------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------



