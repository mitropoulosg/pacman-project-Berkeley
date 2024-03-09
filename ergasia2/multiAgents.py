# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        "*** YOUR CODE HERE ***"
        currentFood = currentGameState.getFood()
        ghostPositions = successorGameState.getGhostPositions() 
        Food = currentFood.asList()      ###We have the position of the food and the ghost positions
        FoodDist=[]      ###We have the distance between the pacman,the food and the ghost
        GhostDist=[]

        for food in Food:
            FoodDist.append(manhattanDistance(food,newPos))
        for ghost in ghostPositions:
            GhostDist.append(manhattanDistance(ghost,newPos))



        for p in GhostDist:
            # If a ghost is at the new position or next to it
            if p < 2:
                # Don't go there
                return(float('-inf'))
            # If there is food in the new position
            elif currentFood[newPos[0]][newPos[1]]:
                # There is no ghost there or next to it,
                # so go there and eat the dot
                return float('inf')
        
        # Pacman is not in immediate danger in the new position, but there is no food there
        # Now estimating the nearest food dot:
        minDist=1/float(min(FoodDist))
        
        
        # We return minDist,
        # and a state near a food dot is more preferable.
        return minDist


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth): #Max_Val is called for the states that Pacman is in
            Actions=gameState.getLegalActions(0)
            #If we are at a terminal state call the evaluation function to assign a value
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:     ###The trvial situations(state)
                return(self.evaluationFunction(gameState),None)
            v_node=-(float("inf"))     ###We are trying to implement the 2 sides of the minimax algorithm the max and the min
            next_Act=None
            #For every successor state, call the Min_Val function to get its value and return the action of the successor state with the greatest value
            for action in Actions:    
                succesor=gameState.generateSuccessor(0,action)      ###In that way that the 2 functions are calling each other is like building the tree(diagrams from tha class)
                sucsValue=min_value(succesor,1,depth)[0]         #We have the available moves and we are seeking for the "best" one                                                              
                if(sucsValue>v_node):            #Here we have as start -infinite
                    v_node,next_Act=sucsValue,action
            return(v_node,next_Act)

        def min_value(gameState,agentID,depth):  #Min_Val is called for the ghost states
            Actions=gameState.getLegalActions(agentID)
            if len(Actions) == 0:
                return(self.evaluationFunction(gameState),None)
            v_node=float("inf")    ###As we see in contrast with max we begin from +infinte
            next_Act=None
            for action in Actions:
                succesor=gameState.generateSuccessor(agentID,action)     
                if(agentID==gameState.getNumAgents() -1):  #If it is the last agent we have, this means every other ghost has received its Minimax value so it's time to call Max_Val so Pacman may continue.  
                    sucsValue=max_value(succesor,depth + 1)[0]
                else:
                    sucsValue=min_value(succesor,agentID+1,depth)[0]        ###We are doing exactly the opposite from the max "function"
                if(sucsValue<v_node):
                    v_node,next_Act=sucsValue,action
            return(v_node,next_Act)
        max_value=max_value(gameState,0)[1]
        return max_value   
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth,alpha,beta): #Max_Val is called for the states that Pacman is in
            Actions=gameState.getLegalActions(0)
            #If we are at a terminal state call the evaluation function to assign a value
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:       ###The trvial situations(state)
                return(self.evaluationFunction(gameState),None)
            v_node=-(float("inf"))                                                                          
            next_Act=None
            #For every successor state, call the Min_Val function to get its value and return the action of the successor state with the greatest value
            for action in Actions:    
                succesor=gameState.generateSuccessor(0,action)                 
                sucsValue=min_value(succesor,1,depth,alpha,beta)[0]   #It is working exactly as the theory of minimax algorithm commands
                if(sucsValue>v_node):                                                                            
                    v_node,next_Act=sucsValue,action
                if (v_node > beta):         #If the value of the successor state is greater than beta there's no need to explore the successors, so we prune them by returning the (v_node, next_action) tuple 
                    return(v_node,next_Act)
                alpha=max(alpha,v_node)  #Alpha value is updated if need be
            return(v_node,next_Act)

        def min_value(gameState,agentID,depth,alpha,beta):  #Min_Val is called for the ghost states
            Actions=gameState.getLegalActions(agentID)
            if len(Actions) == 0:
                return(self.evaluationFunction(gameState),None)
            v_node=float("inf")     ###As we see in contrast with max we begin from +infinte
            next_Act=None
            for action in Actions:
                succesor=gameState.generateSuccessor(agentID,action)     
                if(agentID==gameState.getNumAgents() -1):  #If it is the last agent we have, this means every other ghost has received its Minimax value so it's time to call Max_Val so Pacman may continue.  
                    sucsValue=max_value(succesor,depth + 1,alpha,beta)[0]
                else:
                    sucsValue=min_value(succesor,agentID+1,depth,alpha,beta)[0]     ###We are doing exactly the opposite from the max "function"
                if(sucsValue<v_node):
                    v_node,next_Act=sucsValue,action
                if(v_node < alpha):   #If the value of the succesor state is less than alpha there's no need to explore the successors, so we prune them by returning the (v_node, next_action) tuple
                    return (v_node,next_Act) 
                beta=min(beta,v_node)  #Beta value is updated if need be
            return(v_node,next_Act)
        
        alpha=-(float("inf"))
        beta=(float("inf"))
        max_value=max_value(gameState,0,alpha,beta)[1]
        return max_value  
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth): #Max_Val is called for the states that Pacman is in
            Actions=gameState.getLegalActions(0)
            #If we are at a terminal state call the evaluation function to assign a value
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:             
                return(self.evaluationFunction(gameState),None)
            v_node=-(float("inf"))                                                                               
            next_Act=None
            #For every successor state, call the Min_Val function to get its value and return the action of the successor state with the greatest value
            for action in Actions:    
                succesor=gameState.generateSuccessor(0,action)                 
                sucsValue=min_value(succesor,1,depth)[0]     #It is working exactly as the theory of minimax algorithm commands
                if(sucsValue>v_node):                                                                            
                    v_node,next_Act=sucsValue,action
            return(v_node,next_Act)

        def min_value(gameState,agentID,depth):  #Min_Val is called for the ghost states
            Actions=gameState.getLegalActions(agentID)
            if len(Actions) == 0:
                return(self.evaluationFunction(gameState),None)
            v_node=0                                                    
            next_Act=None
            for action in Actions:
                succesor=gameState.generateSuccessor(agentID,action)     
                if(agentID==gameState.getNumAgents() -1):  #If it is the last agent we have, this means every other ghost has received its Minimax value so it's time to call Max_Val so Pacman may continue.  
                    sucsValue=max_value(succesor,depth + 1)[0]
                else:
                    sucsValue=min_value(succesor,agentID+1,depth)[0]      
                probability=sucsValue/len(Actions)
                v_node+=probability
            return(v_node,next_Act)
        max_value=max_value(gameState,0)[1]
        return max_value   
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    Food=currentGameState.getFood()
    Capsules=currentGameState.getCapsules()
    FoodLst=Food.asList()
    Ghosts=currentGameState.getGhostStates() 

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    foodDistances=[]
    maxFDist = 1
    for food in FoodLst:
        foodDistances.append(manhattanDistance(food,position))
    if len(foodDistances)>0: #The best value we need is the reciprocal of the furthest food state
        maxFDist= 1/float(max(foodDistances))

    GhostDistances=[]  
    maxGhDist =1         # we have take into account more parameters in order to have a better evalution function   
    for ghost in Ghosts:
        if ghost.scaredTimer==0:                                                               
            GhostDistances.append(manhattanDistance(position,ghost.getPosition()))
    if len(GhostDistances) > 0:
        maxGhDist=(max(GhostDistances))  

    ScaredGhostDistances=[]  
    ScaredmaxGhDist =1         
    for ghost in Ghosts:
        if ghost.scaredTimer>0:                                                                                     
            ScaredGhostDistances.append(manhattanDistance(position,ghost.getPosition()))
    if len(ScaredGhostDistances) > 0:
        ScaredmaxGhDist=1/float(max(ScaredGhostDistances))     
    CapsuleDistances=[]   
    maxCapDist=1                 
    for capsule in Capsules:        
        CapsuleDistances.append(manhattanDistance(position,capsule))
    if len(CapsuleDistances) > 0:
        maxCapDist=1/float(max(CapsuleDistances))      
    
    score=currentGameState.getScore() + maxCapDist - maxGhDist + maxFDist + ScaredmaxGhDist
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
