import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
from sklearn import svm  # provides Support Vector Regression
import csv
import math
import sys

# Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class FitnessAnalyzer:
    def __init__(self):
        self.fitnessdata = FromFinessFileMLR.FitnessResults()
    # ------------------------------------------------------------------------------
    def getAValidrow(self, numOfFea, eps=0.015):
        sum = 0
        while (sum < 3):
            V = zeros(numOfFea)
            for j in range(numOfFea):
                r = random.uniform(0, 1)
                if (r < eps):
                    V[j] = 1
                else:
                    V[j] = 0
            sum = V.sum()
        return V
    # ------------------------------------------------------------------------------
    def Create_A_Population(self, numOfPop, numOfFea):
        population = random.random((numOfPop, numOfFea))
        for i in range(numOfPop):
            V = self.getAValidrow(numOfFea)
            for j in range(numOfFea):
                population[i][j] = V[j]
        return population
    # ------------------------------------------------------------------------------
    # The following creates an output file. Every time a model is created the
    # descriptors of the model, the ame of the model (ex: "MLR" for multiple
    # linear regression of "SVM" support vector machine) the R^2 of training, Q^2
    # of training,R^2 of validation, and R^2 of test is placed in the output file

    def createAnOutputFile(self):
        file_name = None
        algorithm = None

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if ((file_name == None) and (algorithm != None)):
            file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__,
                                                    algorithm.model.__class__.__name__, algorithm.gen_max, timestamp)
        elif file_name == None:
            file_name = "{}.csv".format(timestamp)
        fileOut = open(file_name, 'w')
        fileW = csv.writer(fileOut)

        fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

        # strform = "utf-8"

        # fileW.writerow([bytes('Descriptor ID', strform), bytes('Fitness', strform), bytes('Model', strform),
        #                bytes('R2', strform), bytes('Q2', strform), bytes('R2Pred_Validation', strform),
        #                bytes('R2Pred_Test', strform)])

        return fileW
    # -------------------------------------------------------------------------------------------
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        #   NewPopulation = create a 2D array of (numOfPop by num of features)
        #   sort the OldPopulation and their fitness value based on the asending
        #   order of the fitness. The lower is the fitness, the better it is.
        #   So, Move two rows with of the OldPopulation with the lowest fitness
        #   to row 1 and row 2 of the new population.
        #
        #   Name the first row to be Dad and the second row to be mom. Create a
        #   one point or n point cross over to create at least couple of children.
        #   children should be moved to the third, fourth, fifth, etc rows of the
        #   new population.
        #   The rest of the rows should be filled randomly the same way you did when
        #   you created the initial population.

        NewPopulation = ndarray((numOfPop, numOfFea))

        BestFitnessIndex = 0
        f_index = 0
        fitness_range = numOfPop

        # Moving the two OldPopulation rows with best fitness to the first and second rows of NewPopulation
        for n in range(0, 2):
            for f_index in range(fitness_range):
                # If fitness at this BestFitnessIndex > fitness at nth position, then nth position is new BestFitnessIndex
                if fitness[BestFitnessIndex] > fitness[f_index]:
                    BestFitnessIndex = f_index
            delete(fitness, f_index)
            # Deep-copying all column elements in a best-fitting OldPopulation row to the current NewPopulation row
            for ni in range(0, numOfFea):
                NewPopulation[n][ni] = OldPopulation[BestFitnessIndex][ni]
            fitness_range -= 1

        # Designating Dad and Mom rows
        Dad = NewPopulation[0]
        Mom = NewPopulation[1]

        # Child at position 2
        for d in range(0, int(numOfFea / 2)):
            NewPopulation[2][d] = Dad[d]
        for m in range(int(numOfFea / 2), numOfFea):
            NewPopulation[2][m] = Mom[m]

        # Child at position 3
        for m in range(0, int(numOfFea / 2)):
            NewPopulation[3][m] = Mom[m]
        for d in range(int(numOfFea / 2), numOfFea):
            NewPopulation[3][d] = Dad[d]

        # Child at position 4
        for i in range(0, numOfFea):
            if i % 2 == 0:
                NewPopulation[4][i] = Mom[i]
            else:
                NewPopulation[4][i] = Dad[i]

        # Child at position 5
        for i in range(0, numOfFea):
            if i % 2 == 0:
                NewPopulation[5][i] = Dad[i]
            else:
                NewPopulation[5][i] = Mom[i]

        # All remaining rows
        for p in range(6, numOfPop):
            V = self.getAValidrow(numOfFea)
            for f in range(0, numOfFea):
                NewPopulation[p][f] = V[f]

        return NewPopulation
    # -------------------------------------------------------------------------------------------
    def PerformOneMillionIteration(self, numOdPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 1
        OldPopulation = population
        while (NumOfGenerations < 1000):
            population = self.createANewPopulation(numOdPop, numOfFea, OldPopulation, fitness)
            fittingStatus, fitness = self.fitnessdata.validate_model(model, fileW, population, TrainX, TrainY,
                                                                      ValidateX, ValidateY, TestX, TestY)
            NumOfGenerations = NumOfGenerations + 1
            print(NumOfGenerations)
        return
    # --------------------------------------------------------------------------------------------

def main():
    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()
    filedata = FromDataFileMLR.DataFromFile()
    fitnessdata = FromFinessFileMLR.FitnessResults()
    analyzer = FitnessAnalyzer()

    # create an output file. Name the object to be FileW
    fileW = analyzer.createAnOutputFile()

    # Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 385

    # we continue exhancing the model; however if after 1000 iteration no
    # enhancement is done, we can quit
    unfit = 1000

    # Final model requirements: The following is used to evaluate each model. The minimum
    # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
    # test should be 0.5
    R2req_train = .6
    R2req_validate = .5
    R2req_test = .5

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = filedata.getAllOfTheData()
    TrainX, ValidateX, TestX = filedata.rescaleTheData(TrainX, ValidateX, TestX)

    fittingStatus = unfit
    population = analyzer.Create_A_Population(numOfPop, numOfFea)
    fittingStatus, fitness = fitnessdata.validate_model(model, fileW, population,
                                                              TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    analyzer.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)


# main routine ends in here
# ------------------------------------------------------------------------------
main()
# ------------------------------------------------------------------------------


