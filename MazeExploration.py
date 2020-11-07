# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:22:38 2020

@author: Prthamesh
"""

import os # os module
current_working_directory = os.getcwd() # get current working directory to store data files that will be generated.
import time # time module is used for benchmarking.
import random # random module is used for maze generation and fire advancement.
import numpy as np # numpy is only used to display the maze on console in displayMaze function.
from copy import deepcopy as dc # deepcopy is used to create copies of the maze.
import pandas as pd # pandas is used to read csv files of generated data and use those dataframes for plotting the line charts.
import plotly.graph_objects as go # plotly library is used for visualization of the data.

# Priority queue is used by A* search and Uniform cost search
class PriorityQueue:
    def __init__(self):
        self.queue = [] # queue is a list
        self.table = {} # associated table to check if queue has a particular element
    
    def insert(self,node):
        self.queue.append(node) # insert node at the end of a queue
        self.table[node[0]] = 1 # make the input state explored
        
    def remove(self):
        node = self.queue[0] # initialize node to first element in a list
        for n in self.queue[1:]: # iterate and fetch the node with least evaluation cost
            if (n[2]+n[3]) < (node[2]+node[3]):
                node = n
        self.queue.remove(node) # remove node with least evaluation cost from the queue
        self.table.pop(node[0]) # pop the same node from table
        return node # return the node

    def has(self,position): # check if state is already in queue
        val = self.table.get(position,0)
        if val == 1: # if state is already in a queue
            return True
        else: # state is not in a queue
            return False
    
    def isEmpty(self): # check if queue is empty
        if len(self.queue) == 0:
            return True
        else:
            return False
        
# Hash table is used to store explored states by search algorithms
class HashTable:  
    def __init__(self):
        self.table = {} # empty dictionary
    
    def insert(self,position):
        self.table[position] = 1 # set the value of position to 1 indicating that its explored
    
    def has(self,position):
        val = self.table.get(position,0)
        if val == 1: # if position is explored
            return True
        else: # position is not explored
            return False

# function to display the maze in a 2-dimensional grid form in a console
def displayMaze(maze,dim): 
    l = [[maze[(r,c)] for c in range(dim)] for r in range(dim)] # convert maze from dictionary form into 2D list form
    arr = np.array(object = l) # make 2D numpy ndarray of the list
    print(arr) # print ndarray

# function to start fire in one of the randomly chosen open cells (excluding start and goal cells).
def startFire(maze,dim):
    rowNums = list(range(dim)) # create a list of row numbers from 0 to dim-1 for random selection
    colNums = list(range(dim)) # create a list of col numbers from 0 to dim-1 for random selection
    while True:
        r,c = random.choice(rowNums) , random.choice(colNums) # make random choice of row and column
        if (r,c) == (0,0) or (r,c) == (dim-1,dim-1) or maze[(r,c)]==1: # if chosen position is start cell / goal cell / blocked cell,
            continue  # then make a random choice again.
        if maze[(r,c)] == 0: # if chosen cell is open
            maze[(r,c)] = 2 # set the cell on fire
            return 

# function to generate and return the maze
def generateMaze(dim,p_blocked):
    maze = {} # maze is a dictionary of the form (rowNum,colNum) : status.
    # Possible values of status are - 0 (Open) , 1 (Blocked) , 2 (On Fire)
    p_weights_ob = [(1-p_blocked)*100 , p_blocked*100] # weights of open and blocked cells for random choice
    for r in range(dim): # iterate through every row
        for c in range(dim): # iterate through every column
            # set the status of a cell at (r,c) to a random choice from 0 (Open) and 1(Blocked)
            maze[(r,c)] = random.choices([0,1],p_weights_ob)[0] 
    maze[(0,0)] = 0 # set the start cell to 0 (Open)
    maze[(dim-1,dim-1)] = 0 # set the goal cell to 0 (Open)
    startFire(maze,dim) # function call to start fire in a randomly selected open cell
    return maze # return the generated maze with initial fire

# function to get Manhattan distance of position from goal position to be used as a heuristic in A* search.
def getManhattanDistance(position,goalPosition):
    return abs(goalPosition[0]-position[0]) + abs(goalPosition[1]-position[1]) # position is of the form (row,col)

# function to get a list of actions available in the current position of an agent in a maze
def getAvailableActions(maze,dim,position):
    actions_list = [] 
    # possible actions are (-1,0) - up , (1,0) - down , (0,-1) - left , (0,1) - right
    # action numbers can simply be added to the current position numbers to get new position numbers.
    r,c = position # unpack the position tuple into row number and column number
    if (r-1)>=0 and maze[(r-1,c)]==0: # check if the cell above exists and is open
        actions_list.append((-1,0)) # action represents up
    if (r+1)<dim and maze[(r+1,c)]==0: # check if the cell below exists and is open
        actions_list.append((1,0)) # action represents down
    if (c-1)>=0 and maze[(r,c-1)]==0: # check if the cell to the left exists and is open
        actions_list.append((0,-1)) # action represents left
    if (c+1)<dim and maze[(r,c+1)]==0: # check if the cell to the right exists and is open
        actions_list.append((0,1)) # action represents right
    return actions_list # return the list of actions available in a given position

# function to get a list of available actions to reach fire.
# This is used to validate the maze.
def getAvailableActionsToReachFire(maze,dim,position):
    actions_list = [] 
    # possible actions are (-1,0) - up , (1,0) - down , (0,-1) - left , (0,1) - right
    # action numbers can simply be added to the current position numbers to get new position numbers.
    r,c = position # unpack the position tuple into row number and column number
    if (r-1)>=0 and maze[(r-1,c)]!=1: # check if the cell above exists and is not blocked
        actions_list.append((-1,0)) # action represents up
    if (r+1)<dim and maze[(r+1,c)]!=1: # check if the cell below exists and is not blocked
        actions_list.append((1,0)) # action represents down
    if (c-1)>=0 and maze[(r,c-1)]!=1: # check if the cell to the left exists and is not blocked
        actions_list.append((0,-1)) # action represents left
    if (c+1)<dim and maze[(r,c+1)]!=1: # check if the cell to the right exists and is not blocked
        actions_list.append((0,1)) # action represents right
    return actions_list # return the list of actions available in a given position

# function to get a computed sequence of actions from current position to goal position
def getPath(pathDictionary,currentPosition,goalPosition):
    # path dictionary is of the form childPosition (key) : parentPosition (value)
    # childPosition can be used as a key because graph searches are used.
    cp = goalPosition # initialize child position to goal position
    positionsList = [] # path will be a sequence of positions (r,c) to visit to reach the goal
    positionsList.insert(0,cp) # insert goal position into a list of positions
    # As search tree is being traversed up, parent positions will be inserted into the list at the beginning
    while True: 
        parentPosition = pathDictionary[cp] # get the parent position of a child position from a dictionary
        if parentPosition == currentPosition: # if parent position is same as current position
            return positionsList # then return a computed sequence of actions.
        positionsList.insert(0,parentPosition) # insert parent position at the beginning of a list
        cp = parentPosition # parent position becomes child position
        
# function to get a dictionary containing counts of burning neighbors for every cell
def getCountOfNeighborsOnFire(maze,dim):
    d = {} # create empty dictionary
    for r in range(dim): # iterate through every row
        for c in range(dim): # iterate through every column
            count = 0 # initialize count of burning neighbors to 0
            if r-1>=0 and maze.get((r-1,c),0)==2: # if cell above is on fire
                count += 1 # increment count by 1
            if r+1<dim and maze.get((r+1,c),0)==2: # if cell below is on fire
                count += 1 # increment count by 1
            if c-1>=0 and maze.get((r,c-1),0)==2: # if cell to the left is on fire
                count += 1 # increment count by 1
            if c+1<dim and maze.get((r,c+1),0)==2: # if cell to the right is on fire
                count += 1 # increment count by 1
            d[(r,c)] = count # set count of burning neighbors of position (r,c)
    return d # return a dictionary of counts of burning neighbors of all cells

# function to search path to a goalPosition. A* search algorithm.
def searchPath(maze,dim,currentPosition,goalPosition):
    pathDictionary = {} # dictionary to store childPosition:parentPosition mapping. It will be used to construct a sequence of actions.
    frontier = PriorityQueue() # Priority queue is required for A* search to order states based on evaluation cost.
    explored = HashTable() # explored stores already visited states (graph search)
    pathCost = 0 # path cost of initial state is 0.
    heuristicCost = getManhattanDistance(currentPosition,goalPosition) # get heuristic cost of initial state
    initialState = (currentPosition,(None,None),pathCost,heuristicCost) # create the initial state in a search tree.
    frontier.insert(initialState) # push the initial state into frontier.
    pathDictionary[currentPosition] = (None,None) # set the parent position of initial position to (None,None)
    while True: 
        if frontier.isEmpty(): # If no more states to be expanded
            return (False,None) # return failure, no path found
        state = frontier.remove() # remove state from a frontier to expand
        explored.insert(state[0]) # add that state to explored
        if state[0] == goalPosition: # if position of removed state is a goal position
            return (True,getPath(pathDictionary,currentPosition,goalPosition)) # compute and return a sequence of actions.
        for action in getAvailableActions(maze,dim,state[0]): # iterate through each action available in state[0] position
            position = (state[0][0]+action[0],state[0][1]+action[1]) # create child position
            pathCost = state[2] + 1 # path cost of child = path cost of parent + 1. Each step costs 1.
            heuristicCost = getManhattanDistance(position,goalPosition) # get heuristic cost of child position.
            if (not frontier.has(position)) and (not explored.has(position)): # if child position is neither in frontier nor in explored
                frontier.insert((position,state[0],pathCost,heuristicCost)) # push child state into the frontier.
                pathDictionary[position] = state[0] # set childPosition:parentPosition
    # return (False,None) 
        
# function to search path to a position initially set on fire. Uniform cost search algorithm.
def searchPathToFire(maze,dim,currentPosition):
    pathDictionary = {} # dictionary to store childPosition:parentPosition mapping. It will be used to construct a sequence of actions.
    frontier = PriorityQueue() # Priority queue is required for Uniform cost search to order states based on path cost.
    explored = HashTable() # explored stores already visited states (graph search)
    pathCost = 0 # path cost of initial state is 0.
    heuristicCost = 0 # setting heuristic cost to 0 makes the search Uniform cost search. Evaluation cost becomes path cost.
    initialState = (currentPosition,(None,None),pathCost,heuristicCost) # create the initial state in a search tree.
    frontier.insert(initialState) # push the initial state into frontier.
    pathDictionary[currentPosition] = (None,None) # set the parent position of initial position to (None,None)
    while True:
        if frontier.isEmpty(): # If no more states to be expanded
            return (False,None) # return failure, no path found
        state = frontier.remove() # remove state from a frontier to expand
        explored.insert(state[0]) # add that state to explored
        if maze[state[0]] == 2: # if removed position is on fire
            return (True,getPath(pathDictionary,currentPosition,state[0])) # compute and return a sequence of actions.
        for action in getAvailableActionsToReachFire(maze,dim,state[0]): # iterate through each action available in state[0] position
            position = (state[0][0]+action[0],state[0][1]+action[1]) # create child position
            pathCost = state[2] + 1 # path cost of child = path cost of parent + 1. Each step costs 1.
            heuristicCost = 0 # setting heuristic cost to 0 makes the search Uniform cost search. Evaluation cost becomes path cost.
            if (not frontier.has(position)) and (not explored.has(position)): # if child position is neither in frontier nor in explored
                frontier.insert((position,state[0],pathCost,heuristicCost)) # push child state into the frontier.
                pathDictionary[position] = state[0] # set childPosition:parentPosition
    # return (False,None)

# function to get the number of reachable open cells by using breadth-first search
def getNumberOfReachableOpenCells(maze,dim):
    currentPosition = (0,0) # start cell
    count = 0 # initialize count of reachable open cells to 0.
    explored = HashTable() # hash table to store visited states (graph search)
    frontier = [] # FIFO queue to store nodes to be expanded in breadth-first search
    frontier.append(currentPosition) # push current position into the frontier
    while frontier: # iterate while frontier is not empty
        position = frontier.pop(0) # remove first state from frontier (FIFO queue)
        explored.insert(position) # insert the state into explored
        count += 1 # increment the count of reachable open cells by 1.
        for action in getAvailableActions(maze,dim,position): # iterate through the list of actions available in position
            childPosition = (position[0]+action[0],position[1]+action[1]) # create a child position
            if (childPosition not in frontier) and (not explored.has(childPosition)): # if child position is neither in frontier nor in explored
                frontier.append(childPosition) # push child position into frontier
    return count # len(explored.table) 

# function to advance fire positions and get a list of newly burning cell positions
def getAdvancedFirePositions(maze,dim,flammability_rate):
    countsDict = getCountOfNeighborsOnFire(maze,dim) # get the counts of burning neighbors of every cell
    advancedPositions = [] # initialize the list of advanced positions to empty list
    for r in range(dim): # iterate through every row 
        for c in range(dim): # iterate through every column
            if (r==0 and c==0) or (r==(dim-1) and c==(dim-1)): # skip the start and goal cells
                continue
            if maze[(r,c)]==1 or maze[(r,c)]==2: # skip the blocked and already burning cells
                continue
            count = countsDict[(r,c)] # get the count of burning neighbors of a cell at (r,c)
            if count == 0: # skip the cells with count 0
                continue
            p_fire = 1 - (1-flammability_rate)**count # compute probability that a cell will be on fire
            p_weights_of = [(1-p_fire)*100,p_fire*100] # weights of a cell being open and on fire
            ch = random.choices([0,2],p_weights_of)[0] # make a random choice from 0 and 2 according to weights
            if ch==2: # if choice is 2
                maze[(r,c)] = 2 # set the cell on fire
                advancedPositions.append((r,c)) # append position of a newly burning cell to a list
    return tuple(advancedPositions) # return tuple of the list to avoid in-place changes to fire advancements

# function to compute fire advancements ahead of time.
# computeFireAdvancementList is a factory / closure function that returns a function object getFireAdvancement.
# To avoid direct access to the fire advancements list, the list is being stored with inner function using state retention.
# List in the factory function gets garbage collected as soon as the function call exits.
def computeFireAdvancementList(mazeForFireAdvancement,dim,flammability_rate):
    no_open_cells = getNumberOfReachableOpenCells(mazeForFireAdvancement,dim) # get the number of reachable open cells
    fireAdvancementList = [] # initialize a list of fire advancements to empty list
    noFireAdvancementCount = 0 # set number of consecutive no fire advancements to 0.
    while True:
        prev_no_open_cells = no_open_cells # store no_open_cells into previous to check for fire advancement
        t = getAdvancedFirePositions(mazeForFireAdvancement,dim,flammability_rate) # get a tuple of positions that are newly set on fire
        no_open_cells -= len(t) # decreament no_open_cells by length of tuple
        fireAdvancementList.append(t) # append the tuple of newly burning positions to a list
        if no_open_cells != prev_no_open_cells: # if number of open cells has decremented
            noFireAdvancementCount = 0 # reset counter to 0
        else: # if number of open cells has not changed
            noFireAdvancementCount += 1 # increment counter by 1
        if noFireAdvancementCount == 500: # if no fire advancement has been observed in 500 consecutive steps, 
            break # # then all possible reachable cells are set on fire (much larger probability of this event). stop the fire advancement
    def getFireAdvancement(moveNumber): # inner function
        return fireAdvancementList[moveNumber] # returns fire advancement positions for a given move number.
    return getFireAdvancement # return inner function object

# function to execute strategy 1
def strategy1(maze,dim,getFireAdvancement):
    currentPosition = (0,0) # start cell
    res,path = searchPath(maze,dim,currentPosition,(dim-1,dim-1)) # search path using A* search algorithm.
    if res: # if path is found
        for moveNumber,position in enumerate(path[:]): # iterate through every position in a path
            currentPosition = position # move the agent to new position
            if currentPosition == (dim-1,dim-1): # if current position is same as goal position
                return True
            if maze[currentPosition] == 2: # if agent catches fire
                return False
            t = getFireAdvancement(moveNumber) # get the tuple of new fire positions correpsonding to move number
            for position in t: # iterate through every position in a tuple
                maze[position] = 2 # set the position on fire
            if maze[currentPosition] == 2: # if agent catches fire
                return False
    else: # if path is not found
        return False
    
# function to execute strategy 2
def strategy2(maze,dim,getFireAdvancement):
    currentPosition = (0,0) # start cell
    moveNumber = 0 # initialize move number to 0
    while True:
        res,path = searchPath(maze,dim,currentPosition,(dim-1,dim-1)) # search path using A* search algorithm.
        if res: # if path is found
            currentPosition = path[0] # update current position to the first position in path
            if currentPosition == (dim-1,dim-1): # if current position is same as goal position
                return True 
            if maze[currentPosition] == 2: # if agent catches fire
                return False
            t = getFireAdvancement(moveNumber) # get the tuple of new fire positions correpsonding to move number
            moveNumber += 1 # increment the move number
            for position in t: # iterate through every position in a tuple
                maze[position] = 2 # set the position on fire
            if maze[currentPosition] == 2: # if agent catches fire
                return False
        else: # if path is not found
            return False
        
# function to get the simulated fire advancements for Monte Carlo Simulation.
def advanceFireForMCS(maze,dim,flammability_rate):
    countsDict = {} # dictionary stores count of burning neighbors of each cell
    for r in range(dim): # iterate through all rows
        for c in range(dim): # iterate through all columns
            count = 0 # initialize count to 0
            if r-1>=0 and maze.get((r-1,c),0)==2: # if neighbor above is on fire
                count += 1 # increment count by 1
            if r+1<dim and maze.get((r+1,c),0)==2: # if neighbor below is on fire
                count += 1 # increment count by 1
            if c-1>=0 and maze.get((r,c-1),0)==2: # if neighbor to the left is on fire
                count += 1 # increment count by 1
            if c+1<dim and maze.get((r,c+1),0)==2: # if neighbor to the right is on fire
                count += 1 # increment count by 1
            countsDict[(r,c)] = count # set computed count in a dictionary
    newFirePositionsList = [] # list that stores positions that are newly set on fire
    for r in range(dim): # iterate through each row
        for c in range(dim): # iterate through each column
            if (r==0 and c==0) or (r==(dim-1) and c==(dim-1)): # if start cell or goal cell
                continue # skip them
            if maze[(r,c)]==1 or maze[(r,c)]==2: # if blocked cell or cell on fire
                continue # skip them
            count = countsDict[(r,c)] # get count of burning neighbors of cell at (r,c)
            if count == 0: # if count of burning neighbors is 0
                continue # skip it
            p_fire = 1 - (1-flammability_rate)**count # probability that a cell will be on fire
            p_weights_of = [(1-p_fire)*100,p_fire*100] # weights of open and fire cells
            ch = random.choices([0,2],p_weights_of)[0] # make a random choice between 0 and 2 according to weights
            if ch==2: # if choice is 2
                newFirePositionsList.append((r,c)) # add this new fire position to the list
                maze[(r,c)] = 2 # set the position on fire
    return newFirePositionsList # return the list of positions newly set on fire

# Perform 1 simulation executing the given number of fire advancements
def monte_carlo_simulation(cp,maze,dim,numberOfAdvancements,flammability_rate):  
    n = 1 # set counter indicating number of advancements to 1
    newFirePositionsList = [] # initialize a list of newly burning simulated positions to empty list
    while n <= numberOfAdvancements: # iterate numberOfAdvancements times
        newFirePositionsList = newFirePositionsList + advanceFireForMCS(maze,dim,flammability_rate) # concatenate fire positions list with the new list received.
        n += 1 # increment the counter by 1
    return newFirePositionsList # return a complete list of newly burning simulated fire positions
        
# function to execute strategy 3
def strategy3(maze,dim,getFireAdvancement,flammability_rate):
        currentPosition = (0,0) # start cell
        moveNumber = 0 # initialize move number to 0
        while True:
            numberOfAdvancements = 10 # number of fire advancements to simulate
            futureFirePositionsSimulationCount = {} # dictionary containing how many times different cells were set on fire during simulation.
            # Dictionary does not contain cells that were never set on fire during simulation.
            simulationRunNumber = 1 # initialize simulation number to 1.
            totalSimulations = 10 # total number of simulations to perform.
            # Fire advancements are simulated and paths are found in simulated future mazes.
            while simulationRunNumber <= totalSimulations:
                # list of positions (row,col) that were set on fire in given number of advancements
                futureFirePositionsList = monte_carlo_simulation(currentPosition,dc(maze),dim,numberOfAdvancements,flammability_rate)
                for position in futureFirePositionsList: # iterate through each position
                    if position in futureFirePositionsSimulationCount: # if position is already in a dictionary,
                        futureFirePositionsSimulationCount[position] += 1 # increment its value by 1
                    else: # if position is not already in a dictionary,
                        futureFirePositionsSimulationCount[position] = 1 # set its value to 1
                simulationRunNumber += 1 # increment simulation number
            futureFirePositionsList = None
            countsList = futureFirePositionsSimulationCount.values() # get a list of counts
            sortedCounts = sorted(list(set(countsList))) # get a sorted list of unique counts
            res = False # initialize res to False. res becomes True when path to goal is found.
            i = 0 # index in a sortedCounts list
            while res == False: # iterate while path is not found
                if i>=len(sortedCounts): # if no path is found in all mazes of decreasing fire simulations
                    break # exit the loop
                maze2 = dc(maze) # make a copy of original maze to add simulated fire positions.
                for position,count in futureFirePositionsSimulationCount.items(): # scan through all key,value pairs
                    if count>=sortedCounts[i]:
                        maze2[position] = 2 # add simulated fire position to maze
                res,path = searchPath(maze2,dim,currentPosition,(dim-1,dim-1)) # search path in simulated maze
                if not res: # if path is not found
                    i += 1
            if not res: # if no path is found in all simulated mazes
                res,path = searchPath(maze,dim,currentPosition,(dim-1,dim-1)) # search path in original maze with no simulations.
            if res: # if path is found
                currentPosition = path[0] # update current position to the first position in path
                if currentPosition == (dim-1,dim-1): # if current position is same as goal position
                    return True
                if maze[currentPosition] == 2: # if agent catches fire
                    return False
                t = getFireAdvancement(moveNumber) # get the tuple of new fire positions correpsonding to move number
                moveNumber += 1 # increment the move number
                for position in t: # iterate through every position in a tuple
                    maze[position] = 2 # set the position on fire
                if maze[currentPosition] == 2: # if agent catches fire
                    return False
            else: # if path is not found
                return False
        
# function to check if there is a path from start cell to goal cell
def isMazeSolvable(maze,dim):
    currentPosition = (0,0) # start cell
    res,path = searchPath(maze,dim,currentPosition,(dim-1,dim-1)) # search path using A* search algorithm.
    return res # return the boolean resulf of search

# function to check if there is a path from start cell to a cell initially set on fire
def isTherePathFromStartToFire(maze,dim):
    currentPosition = (0,0) # start cell
    res,path = searchPathToFire(maze,dim,currentPosition) # search path using Uniform cost search algorithm.
    return res # return the boolean result of search
        
# function to compare average successes of 3 strategies against same maze with same fire advancements.
# strategies do not access fire advancements.
def compare(dim,startF,stopF,stepF,totalRuns,s1=0,s2=0,s3=0,p_blocked=0.3):
    flammability_rate = startF # start with flammability rate of 0
    # parent directory where the generated data of comparisons is to be stored
    #parent_directory = r'F:\Fall 2020\Introduction to AI\Assignments\FireMaze\FireMazeProject\CompareData'
    parent_directory = current_working_directory
    header = 'flammability_rate,avg_success,time_taken\n' # header to be written in every file
    if s1:
        f1 = open(parent_directory+'\{}_{}_s1.csv'.format(dim,totalRuns),'w') # f1 stores data of strategy 1
        f1.write(header) # write header to file f1
    if s2:
        f2 = open(parent_directory+'\{}_{}_s2.csv'.format(dim,totalRuns),'w') # f2 stores data of strategy 2
        f2.write(header) # write header to file f2
    if s3:
        f3 = open(parent_directory+'\{}_{}_s3.csv'.format(dim,totalRuns),'w') # f3 stores data of strategy 3
        f3.write(header) # write header to file f3
    
    # repeat while flammability rate is <= 1.0
    while flammability_rate<=stopF: # 1.03 accomodates for floating-point arithmetic errors
        runNumber = 1 # set run number to 1
        s1_success = 0 # set number of successes of strategy 1 to 0
        s2_success = 0 # set number of successes of strategy 2 to 0
        s3_success = 0 # set number of successes of strategy 3 to 0
        s1_time_taken = 0
        s2_time_taken = 0
        s3_time_taken = 0
        # repeat process of maze generation to comparion process totalRuns number of times
        while runNumber <= totalRuns:
            mazeOriginal = generateMaze(dim,p_blocked) # generate maze
            # startFire(mazeOriginal,dim) # start fire in a random open cell except start and goal cell
            tempMaze = dc(mazeOriginal) # copy maze to check if it is solvable and if there is a path from start to fire
            if isMazeSolvable(tempMaze,dim) and isTherePathFromStartToFire(tempMaze,dim):
                runNumber += 1 # if maze is solvable and there is a path from start to fire, increment runNumber by 1
                mazeForFireAdvancement = dc(mazeOriginal) # copy maze to advance fire
                getFireAdvancement = computeFireAdvancementList(mazeForFireAdvancement,dim,flammability_rate)
                if s1:
                    s1_start_time = time.time_ns()
                    res1 = strategy1(dc(mazeOriginal),dim,getFireAdvancement) # get a boolean result of strategy 1
                    s1_end_time = time.time_ns()
                    s1_time_taken += (s1_end_time - s1_start_time)/(10**9)
                    if res1: # if strategy 1 succeeds
                        s1_success += 1 # increment s1_success by 1.
                if s2:
                    s2_start_time = time.time_ns()
                    res2 = strategy2(dc(mazeOriginal),dim,getFireAdvancement) # get a boolean result of strategy 2
                    s2_end_time = time.time_ns()
                    s2_time_taken += (s2_end_time - s2_start_time)/(10**9)
                    if res2: # if strategy 2 succeeds, 
                        s2_success += 1 # increment s2_success by 1.
                if s3:
                    s3_start_time = time.time_ns()
                    res3 = strategy3(dc(mazeOriginal),dim,getFireAdvancement,flammability_rate) # get a boolean result of strategy 3
                    s3_end_time = time.time_ns()
                    s3_time_taken += (s3_end_time - s3_start_time)/(10**9)
                    if res3: # if strategy 3 succeeds,
                        s3_success += 1 # increment s3_success by 1.
                print(runNumber,s1_success,s2_success,s3_success)
        print('Total Runs : {}\nFlammability Rate : {}'.format(totalRuns,flammability_rate))
        flammability_rate = round(flammability_rate,2) # round flammability rate to 2 decimal places
        if s1:
            f1.write('{},{},{}\n'.format(flammability_rate,(s1_success/totalRuns),s1_time_taken)) # write results of strategy 1 to file f1
            print('Strategy 1 successes :',s1_success)
        if s2:
            f2.write('{},{},{}\n'.format(flammability_rate,(s2_success/totalRuns),s2_time_taken)) # write results of strategy 2 to file f2
            print('Strategy 2 successes :',s2_success)
        if s3:
            f3.write('{},{},{}\n'.format(flammability_rate,(s3_success/totalRuns),s3_time_taken)) # write results of strategy 3 to file f3
            print('Strategy 3 successes :',s3_success)
        flammability_rate += stepF # increment flammability rate by 0.1
        print('-'*40)
    if s1:f1.close() # close file f1
    if s2:f2.close() # close file f2
    if s3:f3.close() # close file f3    
    
# function that calls compare function and also measures time taken for generating comparison data
def createData(dim,startF,stopF,stepF,totalRuns,s1=0,s2=0,s3=0):
    start_time = time.time_ns() # start time of comparison
    compare(dim,startF,stopF,stepF,totalRuns,s1,s2,s3) # function call to compute averages
    end_time = time.time_ns() # end time of comparison
    print('Time taken :',(end_time-start_time)/(10**9),'seconds') # print time taken in seconds
  
createData(dim=10,startF=0.1,stopF=1.02,stepF=0.1,totalRuns = 10,s1=1,s2=1,s3=1)

# function that plots line chart of flammability rate vs average success for a given dimension and total 
def plotLineChartAverageSuccessVsFlammability(dim,total_runs,s1=0,s2=0,s3=0):
    parentDirectory = current_working_directory
    # parentDirectory = r'F:\Fall 2020\Introduction to AI\Assignments\FireMaze\FireMazeProject\CompareData'
    fig = go.Figure()
    if s1:
        df1 = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s1.csv')
        fig.add_trace(go.Scatter(x=df1['flammability_rate'],y=df1['avg_success'],mode='lines+markers',line=dict(color='darkgreen'),name='Strategy 1'))
    if s2:
        df2 = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s2.csv')
        fig.add_trace(go.Scatter(x=df2['flammability_rate'],y=df2['avg_success'],mode='lines+markers',line = dict(color='red'),name='Strategy 2'))
    if s3:
        df3 = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s3.csv')
        fig.add_trace(go.Scatter(x=df3['flammability_rate'],y=df3['avg_success'],mode='lines+markers',line = dict(color='black'),name='Strategy 3'))
    fig.update_layout(title="Dimension of maze : "+str(dim) + ' | Total runs : '+str(total_runs),
                      xaxis_title='Flammability Rate',
                      yaxis_title = 'Average Success',
                      legend_title = 'Strategy Names')
    fig.show()
    
plotLineChartAverageSuccessVsFlammability(dim=6,total_runs=100,s1=1,s2=1,s3=1)

def plotLineChartComputationTimeVsDimension(dimRange,flammability_rate=0.1,total_runs=1,s1=0,s2=0,s3=0):
    parentDirectory = current_working_directory
    # parentDirectory = r'F:\Fall 2020\Introduction to AI\Assignments\FireMaze\FireMazeProject\CompareData'
    fig = go.Figure()
    if s1:
        time_taken_list_s1 = []
        for dim in dimRange:
            try:
                df = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s1.csv')
                row = df.loc[df['flammability_rate']==flammability_rate]['time_taken']
                time_taken = row.iloc[0]
                time_taken_list_s1.append(time_taken)
            except FileNotFoundError as e:
                print(e)
        print(dimRange)
        print(time_taken_list_s1)
        df1 = pd.DataFrame(list(zip(dimRange,time_taken_list_s1)),columns=['dimension','time_taken'])
    if s2:
        time_taken_list_s2 = []
        for dim in dimRange:
            try:
                df = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s2.csv')
                row = df.loc[df['flammability_rate']==flammability_rate]['time_taken']
                time_taken = row.iloc[0]
                time_taken_list_s2.append(time_taken)
            except FileNotFoundError as e:
                print(e)
        df2 = pd.DataFrame(list(zip(dimRange,time_taken_list_s2)),columns=['dimension','time_taken'])
    if s3:
        time_taken_list_s3 = []
        for dim in dimRange:
            try:
                df = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s3.csv')
                row = df.loc[df['flammability_rate']==flammability_rate]['time_taken']
                time_taken = row.iloc[0]
                time_taken_list_s3.append(time_taken)
            except FileNotFoundError as e:
                print(e)
        df3 = pd.DataFrame(list(zip(dimRange,time_taken_list_s3)),columns=['dimension','time_taken'])
    if s1:
        fig.add_trace(go.Scatter(x=df1['dimension'],y=df1['time_taken'],mode='lines+markers',line=dict(color='darkgreen'),name='Strategy 1'))
    if s2:
        fig.add_trace(go.Scatter(x=df2['dimension'],y=df2['time_taken'],mode='lines+markers',line = dict(color='red'),name='Strategy 2'))
    if s3:
        fig.add_trace(go.Scatter(x=df3['dimension'],y=df3['time_taken'],mode='lines+markers',line = dict(color='black'),name='Strategy 3'))
    fig.update_layout(title="Computation Time vs Dimension",
                      xaxis_title='Dimension',
                      yaxis_title = 'Computation Time (seconds)',
                      legend_title = 'Strategy Names')
    fig.show()

plotLineChartComputationTimeVsDimension(dimRange=list(range(10,105,10)),
                                        flammability_rate=0.1,
                                        total_runs=1,
                                        s1=1,s2=1,s3=0)

# function that plots line chart of flammability rate vs average success for a given dimension and total 
def plotLineChartAverageSuccessVsDimension(flammability_rate=0.1,total_runs=1,s1=0,s2=0,s3=0):
    parentDirectory = current_working_directory
    # parentDirectory = r'F:\Fall 2020\Introduction to AI\Assignments\FireMaze\FireMazeProject\CompareData'
    fig = go.Figure()
    if s1:
        avgSuccessList = []
        for dim in range(10,110,10):
            df1 = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s1.csv')
            row = df1.loc[df1['flammability_rate']==flammability_rate]['avg_success']
            print(row)
            avgSuccess = row.iloc[0]
            avgSuccessList.append(avgSuccess)
        fig.add_trace(go.Scatter(x=list(range(10,110,10)),y=avgSuccessList,mode='lines+markers',line=dict(color='darkgreen'),name='Strategy 1'))
    if s2:
        df2 = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s2.csv')
        fig.add_trace(go.Scatter(x=df2['flammability_rate'],y=df2['avg_success'],mode='lines+markers',line = dict(color='red'),name='Strategy 2'))
    if s3:
        df3 = pd.read_csv(parentDirectory+'\\'+str(dim)+'_'+str(total_runs)+'_s3.csv')
        fig.add_trace(go.Scatter(x=df3['flammability_rate'],y=df3['avg_success'],mode='lines+markers',line = dict(color='black'),name='Strategy 3'))
    fig.update_layout(title="Dimension of maze : "+str(dim) + ' | Total runs : '+str(total_runs),
                      xaxis_title='Flammability Rate',
                      yaxis_title = 'Average Success',
                      legend_title = 'Strategy Names')
    fig.show()
    
plotLineChartAverageSuccessVsDimension(0.1,1,s1=1,s2=1,s3=1)