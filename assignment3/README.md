# Assignment 3 - KNN and Decision Tree
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

```
pip install -r requirements
```

2. Run our implementation by running the following (pass the -v or --verbose flag to produce verbose output)

```
python knn.py -v 1 # This will output the number of training, testing, and Leave-One-Out errors
```

If you have any questions or problems, let us know. Thanks.

# K-Nearest Neighbor

## KNN Description
To calculate the nearest neighbor solution, we first needed to normalize all the features
for the training and testing data on a range of 0-1. Afterwards, it was simply a matter of
inputing the proper training or testing arguments into the "sklearn" knn solver which would
output classification predictions. To calculate the Leave-One-Out cross validation (LOOCV) error,
we needed to remove one feature input (and corresponding output) from the knn training
data set. By comparing the LOOCV predictions to the actual classifications, we were able to
compute the error or number of mistakes.

## Training, Testing, and LOOCV Results
Running the command "python knn.py -v 1" results in the computation of training, testing,
and LOOCV errors for values of K from 1, 3, 5, ..., 51. The Excel file "knn_data_table.xlsx"
contains a summary of this information, including a plot of each of the errors as a function
of K.

The results display some interesting behavior. For example, when K=1, the training set
predictions are no difference than the original set of training y values (no error).
This is likely because the knn model was trained with the training input features and
correct training classifications, meaning all data would map to its original outputs.
Around K=7, the training and LOOCV errors are slightly rising, however the testing data
meets a low of **15** errors anad 94.72% accuracy. As K increased above K=7, each the training,
testing, and LOOCV errors began to rise at a steady pace with minimul declines.

Based on our test data, our choice of K would be K=7, primarily because the testing
set experienced the fewest errors when K=7. Additionally, the combined sum of the
three values of prediction error tests is still only 32 errors out of 284*3 combined tests
(96.24% accuracy). Nearly all values of K above 7 are have a combined error count
above 40 and the accuracies are steadily decreasing from 95%.

# Decision Tree


## Decision Tree Stump
You can run the decision tree stump by running `python decision_tree.py`.

To find the decision tree stump we found threshold values for each
attribute (using a max sum array algorithm) and then calculated the information
gain for splitting on that threshold for that attribute.

The best information gain we found was for attribute number **22**. We found that
we could get the best information gain if we split attribute **22** at a value
of *115.7* where values below this value were classified as -1 and values above
were classified as 1. The information gain we got from splitting at this value
was **0.4883**.

Splitting upon this attribute at the threshold above gave us a **89.7**
accuracy over testing data and a **93.9%** over training data.

Decision Tree Stump
```
Attribute #: 22   
Information Gain: 0.4883  
Percent Error (Testing): 89.7%
Percent Error (Training): 93.9%
```

## Decision Tree
Unfortunately we weren't able to complete part two of the decision tree!
