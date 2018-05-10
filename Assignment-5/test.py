import time # provides timing for benchmarks
from numpy import * # provides complex math and array functions
#from sklearn import mlr # provides Multiple Linear Regression
import csv
import math
import sys
#Local files created by me
import FromDataFileMLR_DE
import FromFinessFileMLR_DE
import mlr
#------------------------------------------------------
def getTwoDecPoint(x):
    return float("%.2f"%x)
#------------------------------------------------------------------------------
def createAnOutputFile():
    file_name = None
    algorithm = None
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if ( (file_name == None) and (algorithm != None)):
        file_name = "{}_{}_gen{}_{}.csv".format(alg.__class__.__name__,
        alg.model.__class__.__name__, alg.gen_max,timestamp)
    elif file_name==None:
        file_name = "{}.csv".format(timestamp)
        fileOut = file(file_name, 'wb')
        fileW = csv.writer(fileOut)
        fileW.writerow(['Descriptor ID', 'No. Descriptors', 'Fitness', 'Model','R2', 'Q2', \
        'R2Pred_Validation', 'R2Pred_Test','SEE_Train', 'SDEP_Validation', 'SDEP_Test', \
        'y_Train', 'yHat_Train', 'yHat_CV', 'y_validation', 'yHat_validation','y_Test', 'yHat_Test'])
    return fileW
#------------------------------------------------------------------------------
def findFitnessOfARow(model, vector, TrainX, TrainY, ValidateX, ValidateY):
    xi = FromFinessFileMLR_DE.OnlySelectTheOnesColumns(vector)
    X_train_masked = TrainX.T[xi].T
    X_validation_masked = ValidateX.T[xi].T
    Yhat_cv = FromFinessFileMLR_DE.cv_predict(X_train_masked, TrainY, model)
    Yhat_validation = model.predict(X_validation_masked)
    Y_fitness = append(TrainY, ValidateY)
    Yhat_fitness = append(Yhat_cv, Yhat_validation)
    fitness = FromFinessFileMLR_DE.calc_fitness(xi, Y_fitness, Yhat_fitness, c=2)
    return fitness
#------------------------------------------------------------------------------
def equal (row1, row2):
    numOfFea = row1.shape[0]
    for j in range(numOfFea):
        if (row1[j] != row2[j]):
            return 0
            return 1
    #------------------------------------------------------------------------------
def theRowIsUniqueInPop(RowI,V, population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    unique = 1
    for i in range(RowI-1):
        for j in range(numOfFea):
            if (equal (V, population[i])):
    #print "It is not unique - doing it again"
            return (not unique)
    return unique
#------------------------------------------------------------------------------
def getAValidRow(population, eps=0.015):
    numOfFea = population.shape[1]
    sum = 0;
    #The following ensure that at least couple of features are
    #selected in each population
    unique = 0
    while (sum < 3) and (not unique):
    #print "****** trying to find a unique row "
        V = zeros(numOfFea)
    for j in range(numOfFea):
        r = random.uniform(0,1)
        if (r < eps):
            V[j] = 1
        else:
            V[j] = 0
            sum = V.sum()
    return V
#------------------------------------------
    population = zeros((numOfPop,numOfFea))
    for i in range(numOfPop):
        V = getAValidRow(population)
    while (not theRowIsUniqueInPop(i,V, population)):
        V = getAValidRow(population)
    for j in range(numOfFea):
        population[i][j] = V[j]
    return population #------------------------------------------------------ #------------------------------------------------------ #------------------------------------------------------
def crossover(P, V):
    numOfFea = V.shape[0]
    CRrate = 0.8
    U = zeros(numOfFea)
    for j in range(numOfFea):
        R = random.uniform(0, 1)
        if (R < CRrate):
            U[j] = P[j]
        else:
            U[j] = V[j]
        return U #------------------------------------------------------
def findMutationFunction(V1, V2, V3):
    numOfFea = V1.shape[0]
    F = 0.5
    V = zeros(numOfFea)
    for i in range(numOfFea):
        V[i] = V3[i] + (F *(V2[i]- V1[i]))
    return V #------------------------------------------------------
def selectARowFromPopulation(population):
    numOfPop = population.shape[0]
    p = int(random.uniform(1,numOfPop))
    return population[p] #------------------------------------------------------
def selectThreeRandomRows(parentPop):
    numOfPop = parentPop.shape[0]
    numOfFea = parentPop.shape[1]
    V1 = selectARowFromPopulation(parentPop)
    V2 = selectARowFromPopulation(parentPop)
    V3 = selectARowFromPopulation(parentPop)
    #The following section ensures that the rows are not the same
    while (equal(V1,V2)):
        V2 = selectARowFromPopulation(parentPop)
    while (equal(V3,V1)) or (equal(V3,V2)):
        V3 = selectARowFromPopulation(parentPop)
    return V1, V2, V3
#------------------------------------------------------
def rowExistInParentPop(V, parentPop):
    numOfPop = parentPop.shape[0]
    for i in range(numOfPop):
        if (equal (V, parentPop[i])):
            return 1
#------------------------------------------------------
def theVecWithMinFitness(fitness, parentPop):
    m = fitness.min()
    numOfPop = parentPop.shape[0]
    for i in range(numOfPop):
        if (fitness[i] == m):
            return parentPop[i]
#------------------------------------------------------
def findNewPopulation(fitness, parentPop, model, \
TrainX, TrainY, ValidateX, ValidateY):
    numOfPop = parentPop.shape[0]
    numOfFea = parentPop.shape[1]
    population = zeros((numOfPop, numOfFea))
    V = theVecWithMinFitness(fitness, parentPop)
    for j in range(numOfFea):
        population[0][j] = V[j]
    for i in range(1, numOfPop):
        U = zeros(numOfFea)
    while ((U.sum()<3) or (not theRowIsUniqueInPop(i,U, population)) \
    or (rowExistInParentPop(U, parentPop))):
        V1, V2, V3 = selectThreeRandomRows(parentPop)
        V = findMutationFunction(V1, V2, V3)
        U = crossover(parentPop[i], V)
        fitnessU = findFitnessOfARow(model, U, TrainX, TrainY, ValidateX, ValidateY)
    if (fitnessU < fitness[i]):
        for j in range(numOfFea):
            population[i][j] = U[j]
    else:
        for j in range(numOfFea):
            population[i][j] = parentPop[i][j]
    return population
#------------------------------------------------------
def getParentPopulation(population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    parentPop = zeros((numOfPop, numOfFea))
    for i in range(numOfPop):
        for j in range(numOfFea):
            parentPop[i][j]= population[i][j]
    return parentPop
#------------------------------------------------------
def checkterTerminationStatus(Times, oldFitness, minimumFitness):
    if (Times == 30):
        print ("***** No need to continue. The fitness not changed in the last 500 generation")
        exit(0)
    elif (oldFitness == minimumFitness):
        Times = Times + 1
    elif (minimumFitness < oldFitness):
        oldFitness = minimumFitness
        Times = 0
        print ("\n***** time is = ", time.strftime("%H:%M:%S", time.localtime()))
        print ("******************** Times is set back to 0 ********************\n")
    return oldFitness, Times
#------------------------------------------------------
def IterateNtimes(model,fileW, fitness, population, parentPop, \
TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    numOfGenerations = 2000
    oldFitness = fitness.min()
    Times = 0
    for i in range(numOfGenerations):
        oldFitness, Times = checkterTerminationStatus(Times, oldFitness, fitness.min())
    
    print ("This is iteration ", i, "Fitness is: ", fitness.min())
    unfit = 1000
    fittingStatus = unfit
    while (fittingStatus == unfit):
        population = findNewPopulation(fitness, parentPop, model, \
        TrainX, TrainY, ValidateX, ValidateY)
        fittingStatus, fitness = FromFinessFileMLR_DE.validate_model(model,fileW, \
        population, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
        parentPop = getParentPopulation(population)
    return
#------------------------------------------------------
# #main program starts in here
# def main():
#     fileW = createAnOutputFile()
#     model = mlr.MLR()
#     numOfPop = 50 # should be 50 population
#     numOfFea = 396 # should be 396 descriptors
#     unfit = 1000
#     # Final model requirements
#     R2req_train = .6
#     R2req_validate = .5
#     R2req_test = .5
#     TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR_DE.getAllOfTheData()
#     TrainX, ValidateX, TestX = FromDataFileMLR_DE.rescaleTheData(TrainX, ValidateX, TestX)
#     unfit = 1000
#     fittingStatus = unfit
#     print ("time is = ", time.strftime("%H:%M:%S", time.localtime()))
#     while (fittingStatus == unfit):
#         population = createInitPopMat(numOfPop, numOfFea)
#         fittingStatus, fitness = FromFinessFileMLR_DE.validate_model(model,fileW, population,
#         TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#         parentPop = getParentPopulation(population)
#     print ("time is = ", time.strftime("%H:%M:%S", time.localtime()))
#     IterateNtimes(model,fileW, fitness, population, parentPop, \
#     TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#     # end of main program

#------------------------------------------------------
# calling the main function to get started
main()
#------------------------------------------------------
#------------------------------------------------------
def CreateInitialLocalBestMatrix(population):
    numOfPop = population.shape[0]
    numOfFea = population.shape[1]
    localBestMatrix = zeros((numOfPop, numOfFea))
    for i in range(numOfPop):
        for j in range(numOfFea):
            localBestMatrix[i][j] = population[i][j]
    return localBestMatrix
#------------------------------------------------------
def CreateInitialLocalBestFitness(fitness):
    numOfPop = fitness.shape[0]
    localBestFitness = zeros(numOfPop)
    for i in range(numOfPop):
        localBestFitness[i] = fitness[i]
    return localBestFitness
#------------------------------------------------------
#main program starts in here
def main():
    fileW = createAnOutputFile()
    model = mlr.MLR()
    numOfPop = 50 # should be 50 population
    numOfFea = 396 # should be 396 descriptors
    unfit = 1000
    # Final model requirements
    R2req_train = .6
    R2req_validate = .5
    R2req_test = .5
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = FromDataFileMLR_DE_BPSO.getAllOfTheData()
    TrainX, ValidateX, TestX = FromDataFileMLR_DE_BPSO.rescaleTheData(TrainX, ValidateX, TestX)
    velocity = createInitVelMat(numOfPop, numOfFea)
    unfit = 1000
    fittingStatus = unfit
    59
    print ("********** time is = ", time.strftime("%H:%M:%S", time.localtime()))
    while (fittingStatus == unfit):
        population = createInitPopMat(numOfPop, numOfFea)
        fittingStatus, fitness = FromFinessFileMLR_DE_BPSO.validate_model(model,fileW, population,
        TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
        globalBestRow = InitializeGlobalBestRow(population[0])
        globalBestFitness = fitness[0]
        globalBestRow, globalBestFitness = findGlobalBest(population,fitness,
        globalBestRow,globalBestFitness)
        localBestMatrix = CreateInitialLocalBestMatrix(population)
        localBestFitness = CreateInitialLocalBestFitness(fitness)
        parentPop = getParentPopulation(population)
    print ("Starting the Loop - time is = ", time.strftime("%H:%M:%S", time.localtime()))
    IterateNtimes(model, fileW, fitness, velocity, population, parentPop,
    localBestFitness,localBestMatrix, globalBestRow,
    globalBestFitness, TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#------------------------------------------------------
main()
#main program ends in here
