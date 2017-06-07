# Assignment 2 - Logistic Regression
- Miles Curry **(currymi)**
- CS434
- Dr. Xiaoli Fern
- Spring 2017


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

# MDP

Entry (j, k) in that matrix (where j is the row and k is the column) gives
the probability of a transition from state j to state k given action i.

1.A (finite) set of states S
2.A (finite) set of actions A
3.Transition Model: 𝑇(𝑠,𝑎,𝑠′) = 𝑃( 𝑠′ | 𝑎,𝑠 )
4.Reward Function: 𝑅(𝑠)
