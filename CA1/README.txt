
Thomas Rex Greenway, 201198319

This is the README File for COMP527 - Assignment 1's source code Perceptron.py and MultiClass_Perceptron.py.

Required libraries: MatPlotLib (3.3.2) and Numpy (1.19.2).
[Programs ran in a base conda environment (conda 4.9.2) in Visual Studio Code]

To run both Perceptron.py and MultiClass_Perceptron.py please ensure that the required libraies are installed
as well as ensure that the relevant data files ('train.data' and 'test.data') are located within the same
directory as the python files. 

Both files should be run as the main program to produce results.

Perceptron.py will run 50 runs of 20 iterations of the binary classification problem on the class pairs (1, 2),
(2, 3), and (1, 3) - producing a graph of training errors across iterations with final training error averages and
testing error averages.

MultiClass_Perceptron.py will run 50 runs of 20 iterations of the multi-class classifier method 'one-vs-all' on
classes 1, 2, and 3 - producing a graph of training errors across iterations with final training error averages and
testing error averages. This program can be performed using L2 regularisation or not. To select L2 regularisation
set variables in final if statement as follows:

reg = True
regCoeff = x

where 'x' is the desired regularisation coefficient (The default value is 0.01).
