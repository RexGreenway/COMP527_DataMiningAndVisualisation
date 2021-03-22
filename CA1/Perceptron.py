"""
#######

Thomas Rex Greenway, 201198319

Implementation of a binary Perceptron for COMP527 - Assignment 1 (Problem 2, 3).

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


def selectClasses(class1, class2, *fileArrays):
    """
    Returns NumPy Arrays of data objects with allocation of 1 and -1 to desired classes.

    Parameters
    ----------
    class1 : int [1, 2, 3]
    class2 : int [1, 2, 3]
        Desired classes, represented by their associated integer, to be compared in a binary 
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
            elif row[-1] == class2:
                arr = np.append(row[:-1], -1)
                array = np.vstack((array, arr))
            else:
                pass
        dataArrays.append(array)
    return dataArrays


class Perceptron():
    """
    Perceptron class for the binary classification problem.

    Attributes
    ----------
    weights : NumPy Array
        Weight vector, including bias term w0.

    """
    def __init__(self):
        """
        Initialises the Perceptron class.
        """
        # weights inc bias
        self.weights = np.zeros(5)

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

                    self.weights[:] += y * X[:]          # update weights and bias
                    
            
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
        reals : NumPy Array
            Array of true class labels from the testData
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
            results[rowIndex] = np.sign(a)
        
        reals = testData[:, -1]

        return reals, results
    
    def reset(self):
        """
        Resets the Perceptron weights to 0.
        """
        self.weights[:] = 0


def main(iterations, runs):
    """
    Main function training Perceptron model on pairs of classes present in the data.

    - Produces a graph of errors present in training across each iteration and percentage of errors.
    - Calculates the percentage of incorrect class predictions for each class pair.

    This function computes the averages of these 2 results over a number of runs to 
    compensate for performance differences across various permutations of the objects in
    the datasets.
    """
    # Class pairs list    
    classPairs = [(1, 2), (2, 3), (1, 3)]

    # Establish tracking arrays
    trainErrTot = np.zeros((iterations, len(classPairs)))
    testErrTot = np.zeros(len(classPairs))

    trainNum = np.zeros(len(classPairs))
    testNum = np.zeros(len(classPairs))

    # Running over multiple permutations to find an average.
    for run in range(runs):
        # Grab data and permute (permutation different for each run)
        trainData, testData = convertData("train.data", "test.data")

        # Run through class pair comparisons
        for pair in classPairs:
            # Select correct data objects from permuted array
            train, test = selectClasses(pair[0], pair[1], trainData, testData)

            # Store number of test and train objects
            testNum[classPairs.index(pair)] = len(test)
            trainNum[classPairs.index(pair)] = len(train)

            # Establish perceptron
            p = Perceptron()

            # TRAINING: Grab training errors across iterations and add to total
            trainErr = p.perceptronTrain(train, iterations)
            trainErrTot[:, classPairs.index(pair)] += trainErr[:]

            # TESTING: Compare predictions and reals class labels
            testReals, testPredictions = p.perceptronTest(test)
            testErrTot[classPairs.index(pair)] += len(test) - np.count_nonzero(testPredictions == testReals)

            # Reset Perceptron
            p.reset()

    # Averages
    trainErrAvg = trainErrTot/runs
    testErrAvg = testErrTot/runs

    # Plot training errors
    plt.plot(trainErrAvg)
    plt.xlim(-1, iterations)
    plt.xticks(np.arange(0, iterations, step = 1))
    plt.xlabel("Iterations")
    plt.ylabel("No. of Errors")
    plt.legend(classPairs)
    plt.title(f"Errors Across {iterations} Iterations For Each Class Pair")
    plt.show()

    # Print Training Error %
    print("Average Number of Training Errors (%): ", 100*trainErrAvg[-1]/trainNum)

    # Print testing errors
    print("Average Number of Testing Errors (%): ", 100*(testErrAvg/testNum))




if __name__ == "__main__":

    iterations = 20
    runs = 50

    main(iterations, runs)
