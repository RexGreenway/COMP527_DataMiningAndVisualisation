
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    """
    Perceptron classifier.

    Parameters
    ------------

    Attributes
    -----------

    """
    def __init__(self, trainingData, testData):
        self.trainingData = trainingData
        self.testData = testData

        # weights and bias
        self.weights = np.zeros(4)
        self.bias = 0

    def PerceptronTrain(self, MaxIter):
        """
        Fit training data.

        """
        # Establish iterations over teh training data
        errors = []

        for _ in range(MaxIter):
            
            # Count errors for this iteration
            error = 0

            # For each instance of the dataset
            for row in self.trainingData:
                # Set Variables
                X = row[:-1]
                y = row[-1]

                # Activation score
                a = np.dot(self.weights, X)  + self.bias

                # checking actiavtion score. (Make sure input data moves class labels to +1 and -1)
                if a * y <= 0:
                    # If incorrect classification
                    error += 1

                    self.bias += y                      # update bias
                    self.weights[:] += y * X[:]         # update weights
            
            errors.append(error)
        
        return errors

    def PerceptronTest(self):
        """
        Test on dataset
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
        IMPLEMENT
        """
        self.weights[:] = 0
        self.bias = 0


def convertData(*files):

    rng = np.random.default_rng()
    convertFunc = lambda name: 1. if name == b"class-1" else (2. if name == b"class-2" else 3.)

    ret = []

    for fil in files:
        
        fileArray = np.genfromtxt(fil, delimiter=",", converters={4 : convertFunc})
        fileArray = rng.permutation(fileArray)

        ret.append(fileArray)

    return ret


def selectClasses(class1, class2, *fileArrays):
    """
    """
    ret = []

    for arr in fileArrays:
        array = np.empty((0, 5))
        for row in arr:
            if row[-1] == class1:
                arr = np.append(row[:-1], 1)
                array = np.vstack((array, arr))
            elif row[-1] == class2:
                arr = np.append(row[:-1], -1)
                array = np.vstack((array, arr))
            else:
                pass
        ret.append(array)
    return ret



if __name__ == "__main__":

    


    trainData, testData = convertData("train.data", "test.data")

    errors = []
    testing = []
    real = []

    

    # Class 1 and 2
    train, test = selectClasses(1, 2, trainData, testData)
    perc = Perceptron(train, test)
    errors.append(perc.PerceptronTrain(20))
    testing.append(perc.PerceptronTest())
    real.append(perc.testData[:, -1])
    perc.reset()

    # Class 2 and 3
    train, test = selectClasses(2, 3, trainData, testData)
    perc = Perceptron(train, test)
    errors.append(perc.PerceptronTrain(20))
    testing.append(perc.PerceptronTest())
    real.append(perc.testData[:, -1])
    perc.reset()

    # Class 1 and 3
    train, test = selectClasses(1, 3, trainData, testData)
    perc = Perceptron(train, test)
    errors.append(perc.PerceptronTrain(20))
    testing.append(perc.PerceptronTest())
    real.append(perc.testData[:, -1])
    
    # Plot Errors in training
    for err in errors:
        plt.plot(err)
    plt.xlim(-1, 20)
    plt.xticks(np.arange(0, 20, step = 1))
    plt.show()


    # Errors in testing
    for test, real in zip(testing, real):
        print("\n", test)
        print(real)

        print("Num. Errors: ", 20 - np.count_nonzero(test==real))

    
    

