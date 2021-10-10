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
        pacmanIndex=0
        # 갈 수 있는 곳 중 점수를 최대화하는 곳으로
        for action in move_candidate:
            v=max(v, self.minValue(gameState.generateSuccessor(agentIndex, action), depth, pacmanIndex+1))
            #팩맨 다음 인덱스부터 시작해서 minvalue 탐색

        return v

    def minValue(self, gameState, depth, agentIndex=0):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # 허용 depth까지 왔다. 가장 maxValue를 도출하는 곳으로 가야 함
            return self.evaluationFunction(gameState)

        v = float("inf")
        move_candidate = gameState.getLegalActions(agentIndex)
        agentNum=gameState.getNumAgents()
        pacmanIndex=0
        if agentIndex==agentNum-1:
            for action in move_candidate:
                # 다음 depth로 넘기고 팩맨의 움직임으로
                v=min(v, self.maxValue(gameState.generateSuccessor(agentIndex, action), depth+1, pacmanIndex))

        else:
            for action in move_candidate:
                v = min(v, self.minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1))

        return v


    def Action(self, gameState):
        ####################### Write Your Code Here ################################
        pacmanIndex = 0

        move_candidate = gameState.getLegalActions(pacmanIndex)

        scores = [self.minValue(gameState.generateSuccessor(pacmanIndex, action), 0, pacmanIndex+1) for action in move_candidate]
        # 팩맨 다음 인덱스부터 minvalue 계산 시작해야
        #print(scores)
        bestScore = max(scores)
        Index = [index for index in range(len(scores)) if scores[index] == bestScore]
        get_index = random.choice(Index)

        return move_candidate[get_index]

        #raise Exception("Not implemented yet")

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
