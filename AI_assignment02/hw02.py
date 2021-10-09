#-*- coding: utf-8 -*-
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


## Example Agent
class ReflexAgent(Agent):

    def Action(self, gameState):
        move_candidate = gameState.getLegalActions()

        scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
        bestScore = max(scores)
        Index = [index for index in range(len(scores)) if scores[index] == bestScore]
        get_index = random.choice(Index)

        return move_candidate[get_index]

    def reflex_agent_evaluationFunc(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvalFunc(currentGameState):
    return currentGameState.getScore()


class AdversialSearchAgent(Agent):

    def __init__(self, getFunc='scoreEvalFunc', depth='2'):
        super().__init__()
        self.index = 0
        self.evaluationFunction = util.lookup(getFunc, globals())
        # 평가 함수를 가져오기

        self.depth = int(depth)


######################################################################################

class MinimaxAgent(AdversialSearchAgent):
    """
      [문제 01] MiniMax의 Action을 구현하시오. (20점)
      (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
    """

    def maxValue(self, gameState, depth, agentIndex=0):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # 허용 depth까지 왔다. 가장 maxValue를 도출하는 곳으로 가야 함
            return self.evaluationFunction(gameState)

        v = float("-inf")
        move_candidate = gameState.getLegalActions(agentIndex)
        for action in move_candidate:
            v=max(v, minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex))

        return v

    def minValue(self, gameState, depth, agentIndex=0):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # 허용 depth까지 왔다. 가장 maxValue를 도출하는 곳으로 가야 함
            return self.evaluationFunction(gameState)

        v = float("inf")
        move_candidate = gameState.getLegalActions(agentIndex)
        for action in move_candidate:
            v=max(v, maxValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex))

        return v


    def Action(self, gameState):
        ####################### Write Your Code Here ################################
        pacmanIndex = 0

        raise Exception("Not implemented yet")

        ############################################################################


class AlphaBetaAgent(AdversialSearchAgent):
    """
      [문제 02] AlphaBeta의 Action을 구현하시오. (25점)
      (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
    """

    def Action(self, gameState):
        ####################### Write Your Code Here ################################

        raise Exception("Not implemented yet")

        ############################################################################


class ExpectimaxAgent(AdversialSearchAgent):
    """
      [문제 03] Expectimax의 Action을 구현하시오. (25점)
      (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
    """

    def Action(self, gameState):
        ####################### Write Your Code Here ################################

        raise Exception("Not implemented yet")

        ############################################################################
