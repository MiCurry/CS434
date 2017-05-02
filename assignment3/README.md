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


# Decision Tree

## Decision Tree Stump
To find the decision tree stump we found threshold values for each
attribute (using a max sum array algorithm) and then calculated the information
gain for splitting on that threshold for that attribute.

The best information gain we found was for attribute number **22**. We found that
we could get the best information gain if we split attribute **22** at a value
of *115.7* where values below this value were classified as -1 and values above
were classified as 1. The information gain we got from splitting at this value
was **0.4883**.

Splitting upon this attribute at the threshold above gave us a **89.7%**
accuracy over testing data and a **93.9%** over training data.

Decision Tree Stump
```
Attribute #: 22   
Information Gain: 0.4883  
Percent Error (Testing): 89.7%
Percent Error (Training): 93.9%
```

To find the threshold of each attribute we used a max sub array algorithm. From
that we calculated the entropies of each split and found the information gain.
