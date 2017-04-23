# Assignment 2 - Logistic Regression
- Miles Curry **(currymi)**, Christian Armatas **(armatasc)**
- CS434
- Dr. Xiaoli Fern
- Spring 2017

In this assignment we used Logistic Regression to determine hand written 4's or
hand written 9's. We did this by implementing a Batch Gradient Decent algorithm
that trained a binary logistic regression classifier.

# Running the Code
The code runs a python 2 so it should be simple to run and requires pip
to make sure you have the packages we use installed.
1. First install the requirements using pip
'''
pip install -r requirements
'''
2. Run our implementation by running the following (pass the -v or --verbose flag to produce verbose output)
'''
python logRegress.py -v 2 # This will determine the learning rate!
'''

If you have any questions or problems, let us know. Thanks.

# 1. Learning Rates And Stopping Condition
For learning rates we determined that the optimal learning rate was **.0011**
which produced a loss of **621.7059**. We decided the best number of iterations to
run was 170.

# 2. Training & Testing Accuracy Vs # of iterations
![Training Rate Accuracy](https://web.engr.oregonstate.edu/~currymi/training_accuracy.png)

Over the majority of iterations the accuracy of our tests increased to almost
100%. At the start of the iterations we saw a sharp increase with a sharp
decrease followed by the rapid increase to the max accuracy.

# 3. L2 Regulation
The pusdocode for the batch learning with L2 regulation is as follows:
```
given training examples x[0, 1, .. i] and y[0, 1, .. i]
w = [0, 0, ... 0]
while(iterations > 0):
    d = zeros(256)
    for i in n:
        Y-hat = (1/(1+e^(-w * x[i])) + 1/2 * lambda * ||w||2 Norm
        error = y - y_hat
        d = d + (error * x)

    w = w + (learning_rate * d)
```

## 3.a Plot
![Lambda Accuracy](https://web.engr.oregonstate.edu/~currymi/lambda_accuracy.png)
Plotting the percent correctness with the regulation function and values of
lambda equalling [-10^3, -10^2, .. 10^3] we found that the percent correct
increased as lambda increased towards zero the percent increased. As it got
close to zero, however, the value deceased to half again. However after that the
percent correct started to increase again.
