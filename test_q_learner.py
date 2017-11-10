"""
Test a Q Learner in a navigation problem.  (c) 2015 Tucker Balch

Uses the q_learner.py API (developed by Arnold Yeung) to find the optimal
path in a maze.

2017-11-11: Modified by Arnold Yeung to be compatiable with Python 3
            and my naming conventions
"""

import numpy as np
import random as rand
import time
import math
import q_learner as ql

# print out the map
def printmap(data):
    print("--------------------")
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 0:
                print(" ",end='')
            if data[row,col] == 1:
                print("O",end='')
            if data[row,col] == 2:
                print("*",end='')
            if data[row,col] == 3:
                print("X",end='')
            if data[row,col] == 4:
                print(".",end='')
        print("")
    print("--------------------")

# find where the robot is in the map
def getrobotpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 2:
                C = col
                R = row
    if (R+C)<0:
        print("warning: start location not defined")
    return R, C

# find where the goal is in the map
def getgoalpos(data):
    R = -999
    C = -999
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 3:
                C = col
                R = row
    if (R+C)<0:
        print("warning: goal location not defined")
    return (R, C)

# move the robot according to the action and the map
def movebot(data,oldpos,a):
    testr, testc = oldpos

    # update the test location
    if a == 0: #north
        testr = testr - 1
    elif a == 1: #east
        testc = testc + 1
    elif a == 2: #south
        testr = testr + 1
    elif a == 3: #west
        testc = testc - 1

    # see if it is legal. if not, revert
    if testr < 0: # off the map
        testr, testc = oldpos
    elif testr >= data.shape[0]: # off the map
        testr, testc = oldpos
    elif testc < 0: # off the map
        testr, testc = oldpos
    elif testc >= data.shape[1]: # off the map
        testr, testc = oldpos
    elif data[testr, testc] == 1: # it is an obstacle
        testr, testc = oldpos

    return (testr, testc) #return the new, legal location

# convert the location to a single integer
def discretize(pos):
    return pos[0]*10 + pos[1]

# run the code to test a learner
if __name__=="__main__":

    verbose = False #print lots of debug stuff if True

    # read in the map
    inf = open('testworlds/world01.csv')
    data = np.array([list(map(float,s.strip().split(','))) for s in inf.readlines()])
    originalmap = data.copy() #make a copy so we can revert to the original map later

    startpos = getrobotpos(data) #find where the robot starts
    goalpos = getgoalpos(data) #find where the goal is

    if verbose: printmap(data)

    rand.seed(5)

    learner = ql.qLearner(num_states=100,\
        num_actions = 4, \
        rar = 0.98, \
        radr = 0.9999, \
        verbose=verbose, \
        dyna=0) #initialize the learner

    #each iteration involves one trip to the goal
    for iteration in range(0,1000): 
        steps = 0
        data = originalmap.copy()
        robopos = startpos
        state = discretize(robopos) #convert the location to a state
        action = learner.querySetState(state) #set the state and get first action
        while robopos != goalpos:

            #move to new location according to action and then get a new action
            newpos = movebot(data,robopos,action)
            if newpos == goalpos:
                r = 1 #reward for reaching the goal
            else:
                r = -1 #negative reward for not being at the goal
            state = discretize(newpos)
            action = learner.query(state,r)

            data[robopos] = 4 # mark where we've been for map printing
            data[newpos] = 2 # move to new location
            robopos = newpos # update the location
            if verbose: printmap(data)
            if verbose: time.sleep(1)
            steps += 1

        print(iteration, "," , steps)
printmap(data)
