
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    """
    Perceptron class.

    Parameters
    ------------
    trainingData : NumPy Array
        asdasd
    testData : NumPy Array
        sfsdfd

    Attributes
    ----------
    weights : NumPy Array
        asfd
    bias : float
        asfddas
    
    Notes
    -----
    asdasd

    """
    def __init__(self, trainingData, testData):
        """
        Initialises the Perceptron class with given data.
        """
        self.trainingData = trainingData
        self.testData = testData

        # weights and bias
        self.weights = np.zeros(4)
        self.bias = 0

    def PerceptronTrain(self, MaxIter):
        """
        Fit training data.

        Parameters
        ----------
        MaxIter : int
            awjhkghd

        Returns
        -------
        errors : NumPy Array
            afhdjk
        """
        # Establish iterations over teh training data
        errors = np.zeros(MaxIter)

        for i in range(MaxIter):
            
            # Count errors for this iteration
            error = 0

            # For each instance of the dataset
            for row in self.trainingData:
                # Set Variables
                X = row[:-1]        # add bias into X at position 0 with value x0 = 1??????
                y = row[-1]

                # Activation score
                a = np.dot(self.weights, X)  + self.bias

                # checking actiavtion score. (Make sure input data moves class labels to +1 and -1)
                if a * y <= 0:
                    # If incorrect classification
                    error += 1

                    self.bias += y                      # update bias
                    self.weights[:] += y * X[:]         # update weights
            
            errors[i] = error
        
        return errors

    def PerceptronTest(self):
        """
        Test on dataset

        Returns
        -------
        errors : NumPy Array
            afhdjk
        """
        results = np.zeros(len(self.testData))

        for rowIndex in range(len(self.testData)):
            # Grab instance
            X = self.testData[rowIndex][:-1]

            # Activation Score
            a = np.dot(self.weights, X) + self.bias

            # Store result
            results[rowIndex] = np.sign(a)

        return results
    
    def reset(self):
        """
        Resets the Perceptron to default values.
        """
        self.weights[:] = 0
        self.bias = 0


def convertData(*files):
    """
    Returns list of arrays of data objects converted from provided data files (csv format).
    
    Parameters
    ----------
    *files : str
        File names of desired text files containing data in csv format.

    Returns
    -------
    dataArrays : list of NumPy Arrays
        Arrays 
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
    Returns list of arrays of data objects with allocation of 1 and -1 to desired classes.

    Parameters
    ----------
    class1 : int [1, 2, 3]
    class2 : int [1, 2, 3]
        Desired classes, represented by their associated integer, to be compared in a binary perceptron.
    *fileArrays : NumPy Arrays
        sfadf
    
    Returns
    -------
    dataArrays : list of NumPy Arrays
        asdjfhgj
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



if __name__ == "__main__":

    # Establish Variables
    iterations = 20
    runs = 100
    classPairs = [(1, 2), (2, 3), (1, 3)]

    # Establish tracking arrays
    trainErrTot = np.zeros((iterations, len(classPairs)))
    testErrTot = np.zeros(len(classPairs))

    # Running over multiple permutations to find an average.
    for run in range(runs):
        # Grab data and permute (permutation different for each run)
        trainData, testData = convertData("train.data", "test.data")

        # Run through class pair comparisons
        for pair in classPairs:
            # Select correct data objects from permuted array
            train, test = selectClasses(pair[0], pair[1], trainData, testData)

            # Establish perceptron
            p = Perceptron(train, test)

            # TRAINING: Grab training errors across iterations and add to total
            trainErr = p.PerceptronTrain(iterations)
            trainErrTot[:, classPairs.index(pair)] += trainErr[:]

            # TESTING: Compare predictions and reals class labels
            testPredictions = p.PerceptronTest()
            testReals = p.testData[:, -1]
            testErrTot[classPairs.index(pair)] += 20 - np.count_nonzero(testPredictions == testReals)

            # Reset Perceptron
            p.reset()

    # Averages
    trainErrAvg = trainErrTot/runs
    testErrAvg = testErrTot/runs

    # Plot training errors
    plt.plot(trainErrAvg)
    plt.xlim(-1, 20)
    plt.xticks(np.arange(0, 20, step = 1))
    plt.xlabel("Iterations")
    plt.ylabel("No. of Errors")
    plt.legend(classPairs)
    plt.show()

    # Print testing errors
    print("Average Number of Testing Errors: ", 100*(testErrAvg/20))
