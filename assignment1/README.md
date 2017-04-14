# assignment1
- Miles Curry **(currymi)**
- CS434
- Dr. Xiaoli Fern
- 9 April 2017

In this assignment we used linear regression to calculate the median
value of owner-occupied homes in Boston. We gather training data from
a set of 443 houses (housing_train.txt) and tested our learned weight
vector on a set of 74 houses (housing_test.txt).

We then altered our learned weight vector by adding in dummy variables
and by creating additional features at random to see the result of on
the Sum of Square Error (SSE).

# Running the Code
The code runs a python 2 so it should be simple to run and requires pip
to make sure you have the packages we use installed.
1. First install the requirements using pip
'''
pip install -r requirements
'''
2. Run our implementation by running the following (pass the -v or --verbose flag to produce verbose output)
'''
python linearRegres.py
'''

If you have any questions or problems, let us know. Thanks.

# Results

One dummy variable. 0 Random Variables added
'''
w = [  3.93620631e+01   4.27733574e-02   6.24991478e-03   3.14816354e+00
  -1.68723350e+01   3.69754546e+00   8.09866767e-03  -1.53674556e+00
   3.24131666e-01  -1.58107542e-02  -1.01573219e+00   1.00919488e-02
  -6.19491108e-01]
SSE = 1741.937
'''

%% TODO %%
