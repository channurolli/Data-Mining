import time
from numpy  import *
from sklearn import svm
import csv
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class Fitness:

    def CreateInitialVelocity(self, numOfPop, numOfFea):
        for i in range(numOfPop):
            for j in range(numOfFea):
                self.VelocityM[i][j] = random.random()
    #------------------------------------------------------------------------------
    #Initail population and Initial Local Best matrix
    def createInitialPopulation(self, numOfPop, numOfFea):
        population = random.random((numOfPop,numOfFea))
        for i in range(numOfPop):
            V = self.getAValidrow(numOfFea)
            for j in range(numOfFea):
                population[i][j] = V[j]
        return population
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
    # The following creates an output file.
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
        NewPopulation = ndarray((numOfPop, numOfFea))

        self.alpha -= (0.17 / self.NofIterations)
        p = 0.5 * (1 + self.alpha)
        for i in range(numOfPop):
            for j in range(numOfFea):
                if self.VelocityM[i][j] <= self.alpha:
                    NewPopulation[i][j] = OldPopulation[i][j]
                elif (self.VelocityM[i][j] > self.alpha) & (self.VelocityM[i][j] <= p):
                    NewPopulation[i][j] = self.LocalBestM[i][j]
                elif (self.VelocityM[i][j] > p) & (self.VelocityM[i][j] <= 1):
                    NewPopulation[i][j] = self.GlobalBestRow[j]
                else:
                    NewPopulation[i][j] = OldPopulation[i][j]
        return NewPopulation
    #-------------------------------------------------------------------------------------------
    def FindGlobalBestRow(self):
        IndexOfBest = 0
        numOfPop = self.LocalBestM.shape[0]
        for i in range(numOfPop):
            if (self.LocalBestM_Fit[i] < self.GlobalBestFitness) \
                    & (self.LocalBestM_Fit[i] > 0):
                self.GlobalBestFitness = self.LocalBestM_Fit[i]
                IndexOfBest = i
        copyto(self.GlobalBestRow, self.LocalBestM[IndexOfBest])
    #-------------------------------------------------------------------------------------------
    def UpdateNewLocalBestMatrix(self, NewPopulation, NewPopFitness):
        numOfPop = self.LocalBestM.shape[0]
        for i in range(numOfPop):
                if self.LocalBestM_Fit[i] > NewPopFitness[i]:
                    copyto(self.LocalBestM[i], NewPopulation[i])
    #-------------------------------------------------------------------------------------------
    def UpdateVelocityMatrix(self, NewPop, c1=2, c2=2, inertiaWeight=0.9):
        numOfPop = self.VelocityM.shape[0]
        numOfFea = self.VelocityM.shape[1]
        for i in range(numOfPop):
            for j in range(numOfFea):
                term1 = c1 * random.random() * (self.LocalBestM[i][j] - NewPop[i][j])
                term2 = c2 * random.random() * (self.GlobalBestRow[j] - NewPop[i][j])
                self.VelocityM[i][j]=term1+term2+(inertiaWeight*self.VelocityM[i][j])
    #-------------------------------------------------------------------------------------------
    def PerformOneMillionIteration(self, numOfPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 1
        OldPopulation = population
        while (NumOfGenerations < self.NofIterations):
            population = self.createANewPopulation(numOfPop, numOfFea, OldPopulation, fitness)
            fittingStatus, fitness = self.fitnessdata.validate_model(model,fileW, population, \
                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

            self.UpdateNewLocalBestMatrix(population, fitness)
            self.FindGlobalBestRow()
            self.UpdateVelocityMatrix(population)

            NumOfGenerations = NumOfGenerations + 1
            print(NumOfGenerations)
        return
    #------------------------------------------------------------------------------
    def __init__(self, numOfPop, numOfFea):
        self.filedata = FromDataFileMLR.DataFromFile()
        self.fitnessdata = FromFinessFileMLR.FitnessResults()
        self.NofIterations = 2000
        self.alpha = 0.5
        self.GlobalBestRow = ndarray(numOfFea)
        self.GlobalBestFitness = 10000
        self.VelocityM = ndarray((numOfPop, numOfFea))
        self.LocalBestM = ndarray((numOfPop, numOfFea))
        self.LocalBestM_Fit = ndarray(numOfPop)

#--------------------------------------------------------------------------------------------
#Main program
def main():
    # Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 385

    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()
    filedata = FromDataFileMLR.DataFromFile()
    fitnessdata = FromFinessFileMLR.FitnessResults()
    analyzer = Fitness(numOfPop, numOfFea)

    # create an output file. Name the object to be FileW
    fileW = analyzer.createAnOutputFile()

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
    population = analyzer.createInitialPopulation(numOfPop,numOfFea)
    fittingStatus, fitness = fitnessdata.validate_model(model,fileW, population, \
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    analyzer.CreateInitialVelocity(numOfPop, numOfFea)
    copyto(analyzer.LocalBestM, population) #initializing LocalBestMatrix as the initial population
    copyto(analyzer.LocalBestM_Fit, fitness)
    analyzer.FindGlobalBestRow()

    analyzer.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW, \
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#main routine ends in here
#------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------



