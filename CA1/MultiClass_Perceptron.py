"""
#######

Thomas Rex Greenway, 201198319

Implementation of multiclass one-vs-rest binary Perceptron for COMP527 - Assignment 1 (Problem 4, 5).

#######
"""

import numpy as np
import matplotlib.pyplot as plt

def convertData(*files):
    """
    Returns NumPy Arrays of data objects converted from provided data files (csv format).
    
    Parameters
    ----------
    *files : str
        File names of desired text files containing data in csv format to be converted.

    Returns
    -------
    dataArrays : list of NumPy Arrays
        Arrays containing data objects from provided text files.

    Notes
    -----
    This function dynamically detects, and converts, class label strings (of the form 'class-i') 
    in the data to corresponding integers i. 
    """
    # NumPy random generator
    rng = np.random.default_rng()
    # String conversion for last column
    convertFunc = lambda name: 1. if name == b"class-1" else (2. if name == b"class-2" else 3.)

    # Conversion for file arguments
    dataArrays = []
    for f in files:
        fileArray = np.genfromtxt(f, delimiter=",", converters={-1 : convertFunc})
        fileArray = rng.permutation(fileArray)
        dataArrays.append(fileArray)
    return dataArrays


def selectClass(class1, *fileArrays):
    """
    Returns list of arrays of data objects with allocation of 1 to desired class objects 
    and -1 to all other objects.

    Parameters
    ----------
    class1 : int [1, 2, 3]
        Desired class, represented by their associated integer, to be compared in a binary
        Perceptron.
    *fileArrays : NumPy Arrays
        Names of desired data containing arrays to be converted.
    
    Returns
    -------
    dataArrays : list of NumPy Arrays
        Arrays containing converted data objects.
    """
    dataArrays = []
    for arr in fileArrays:
        array = np.empty((0, 5))
        for row in arr:
            # Allocation of 1 and -1 to desired classes and added to sub array.
            if row[-1] == class1:
                arr = np.append(row[:-1], 1)
                array = np.vstack((array, arr))
            else:
                arr = np.append(row[:-1], -1)
                array = np.vstack((array, arr))
        dataArrays.append(array)
    return dataArrays

class Perceptron():
    """
    Perceptron class for the binary classification problem.

    Parameters
    ------------
    regularistation : boolean (Default = False)
        Weights updated according to L2 regularised rule if True, normal update rule if False.]
    regCoeff : float
        Regularisation coeffecient used in weight update if regularisation is True.

    Attributes
    ----------
    weights : NumPy Array
        Weight vector, including bias term w0.
    """
    def __init__(self, regularisation = False, regCoeff = 1):
        """
        Initialises the Perceptron class.
        """
        self.regularisation = regularisation
        self.regCoeff = regCoeff

        # weights inc bias
        self.weights = np.zeros(5, dtype=np.float64)

    def perceptronTrain(self, trainingData, MaxIter):
        """
        Adapts Perceptron model weights to fit given training data over given number of iterations.

        Parameters
        ----------
        trainingData : NumPy Array
            Data array of data objects along with their class labels in the final column.
        MaxIter : int
            Number of iterations to complete traing over.

        Returns
        -------
        errors : NumPy Array
            Array containg the number of errors, and thus updates, in each iteration.
        """
        # Establish the

        # Establish iteration error tracker over the training data
        errors = np.zeros(MaxIter)

        for i in range(MaxIter):
            
            # Count errors for this iteration
            error = 0

            # For each instance of the dataset
            for row in trainingData:
                # Set Variables (with bias term at index 0 for input object)
                X = np.insert(row[:-1], 0, [1])
                y = row[-1]

                # Activation score
                a = np.dot(self.weights, X)

                # checking actiavtion score. (Make sure input data moves class labels to +1 and -1)
                if a * y <= 0:
                    # Error tracking
                    error += 1
                    
                    # update weights
                    if self.regularisation:
                        self.weights[:] += (2 * self.regCoeff) * self.weights[:] + (y * X[:])
                    else:
                        self.weights[:] += y * X[:]
                    
            
            errors[i] = error
        
        return errors

    def perceptronTest(self, testData):
        """
        Tests a dataset aginst model's developed weights.

        Parameters
        ----------
        testData : NumPy Array
            Data array of data objects along with their class labels in the final column.

        Returns
        -------
        results : NumPy Array
            Array of predicted class labels.
        """
        results = np.zeros(len(testData))

        for rowIndex in range(len(testData)):
            # Grab instance (with bias term at index 0)
            X = testData[rowIndex][:-1]
            X = np.insert(X, 0, [1])

            # Activation Score
            a = np.dot(self.weights, X)

            # Store result
            results[rowIndex] = a

        return results
    
    def reset(self):
        """
        Resets the Perceptron weights to 0.
        """
        self.weights[:] = 0


def main(iterations, runs, reg, regCoeff):
    """
    Main function performing one-vs-rest multi-class classification on 3 classes.

    - Produces graph of errors present in training for each class.
    - Calculates the percentage incorrect class predictions.

    The results computed are the averages over a number of runs to 
    compensate for performance differences across various permutations of the objects in
    the datasets.
    """
    # Class list
    classes = [1, 2, 3]

    # Establish tracking arrays
    trainErrTot = np.zeros((iterations, len(classes)))
    testErrTot = 0

    # Running over multiple permutations to find an average.
    for run in range(runs):
        # Grab data and permute (permutation different for each run)
        trainData, testData = convertData("train.data", "test.data")

        # Array to store all relts to compare
        testNum = len(testData)
        trainNum = len(trainData)
        results = np.zeros((testNum, len(classes)))

        # Run through class pair comparisons
        for c in classes:
            cIndex = classes.index(c)

            # Select correct data objects from permuted array
            train, test = selectClass(c, trainData, testData)

            # Establish perceptron
            p = Perceptron(reg, regCoeff)

            # TRAINING: Grab training errors across iterations and add to total
            trainErr = p.perceptronTrain(train, iterations)
            trainErrTot[:, cIndex] += trainErr[:]

            # TESTING: Compare predictions and reals class labels
            testPredictions = p.perceptronTest(test)
            results[:, cIndex] = testPredictions[:]

            # Reset Perceptron
            p.reset()
        
        # Predicted vs Real class labels
        pred = np.zeros(testNum)
        for objIndex in range(testNum):
            pred[objIndex] = np.argmax(results[objIndex]) + 1       # +1 because class label = (index + 1)
        testReals = testData[:, -1]

        # Total Errors across the runs
        testErrTot += testNum - np.count_nonzero(pred == testReals)

    # Averages
    trainErrAvg = trainErrTot/runs
    testErrAvg = testErrTot/runs

    # Plot training errors
    plt.plot(trainErrAvg)
    plt.xlim(-1, 20)
    plt.xticks(np.arange(0, 20, step = 1))
    plt.xlabel("Iterations")
    plt.ylabel("No. of Errors")
    plt.legend(classes)
    plt.title(f"Errors Across {iterations} Iterations For Each Class")
    plt.show()

    # Print training errors
    print("Average Number of Training Errors (%): ", 100*trainErrAvg[-1]/trainNum)

    # Print testing errors
    print("Average Number of Testing Errors (%): ", 100*(testErrAvg/testNum))





if __name__ == "__main__":

    iterations = 20
    runs = 50
    reg = False
    regCoeff = 0.01

    main(iterations, runs, reg, regCoeff)