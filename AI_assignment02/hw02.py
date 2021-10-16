# -*- coding: utf-8 -*-
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
        # cur game state 에 action을 적용한거
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvalFunc(currentGameState):
    return currentGameState.getScore()


class AdversialSearchAgent(Agent):

    def __init__(self, getFunc='scoreEvalFunc', depth='2'):
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
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            # terminal state
            return self.evaluationFunction(gameState)

        pacmanIndex = 0
        firstGhostIndex = pacmanIndex + 1
        agentNum = gameState.getNumAgents()
        move_candidate = gameState.getLegalActions(agentIndex)
        # 가능한 움직임

        successorValues = [self.minValue(gameState.generateSuccessor(agentIndex, action), depth, firstGhostIndex)
                           for action in move_candidate]
        # max in minValue of every successor
        return max(successorValues)

    def minValue(self, gameState, depth, agentIndex=0):
        # 모든 ghost들의 움직임을 다 완료해야 한다는 것에 유의
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # terminal state
            return self.evaluationFunction(gameState)

        move_candidate = gameState.getLegalActions(agentIndex)
        pacmanIndex = 0
        agentNum = gameState.getNumAgents()

        successorValues = [self.minValue(gameState.generateSuccessor(agentIndex, action), depth,
                                         agentIndex + 1) if agentIndex < agentNum - 1
                           else self.maxValue(gameState.generateSuccessor(agentIndex, action), depth + 1, pacmanIndex)
                           for action in move_candidate]
        # 마지막 유령 즉 index==agentNum-1 이 되면 다음 depth로 넘어가 줘야 한다
        # 그전에는 다음 유령의 행동을 생각해 줘야 한다
        return min(successorValues)

    def Action(self, gameState):
        ####################### Write Your Code Here ################################
        pacmanIndex = 0
        firstGhostIndex = pacmanIndex + 1
        move_candidate = gameState.getLegalActions(pacmanIndex)

        scores = [self.minValue(gameState.generateSuccessor(pacmanIndex, action), 0, firstGhostIndex) for action in
                  move_candidate]
        # 팩맨 다음 인덱스부터 minvalue 계산 시작해야
        # print(scores)
        bestScore = max(scores)
        Index = [index for index in range(len(scores)) if scores[index] == bestScore]
        get_index = random.choice(Index)

        return move_candidate[get_index]

        # raise Exception("Not implemented yet")

        ############################################################################


class AlphaBetaAgent(AdversialSearchAgent):
    """
      [문제 02] AlphaBeta의 Action을 구현하시오. (25점)
      (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
    """

    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # terminal of search
            return self.evaluationFunction(gameState)

        move_candidate = gameState.getLegalActions(agentIndex)
        pacmanIndex = 0
        firstGhostIndex = pacmanIndex + 1
        # 갈 수 있는 곳 중 점수를 최대화하는 곳으로
        successorValues = []
        for action in move_candidate:
            successorValue = self.minValue(gameState.generateSuccessor(pacmanIndex, action), depth, firstGhostIndex,
                                           alpha, beta)
            if successorValue > beta:
                return successorValue
            alpha = max(alpha, successorValue)
            successorValues.append(successorValue)

        return max(successorValues)

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # terminal of search
            return self.evaluationFunction(gameState)

        move_candidate = gameState.getLegalActions(agentIndex)
        agentNum = gameState.getNumAgents()
        pacmanIndex = 0
        firstGhostIndex = pacmanIndex + 1

        successorValues = []
        for action in move_candidate:
            if agentIndex < agentNum - 1:
                successorValue = self.minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1,
                                               alpha, beta)
            else:
                # go to the next depth search if it is the last ghost
                successorValue = self.maxValue(gameState.generateSuccessor(agentIndex, action), depth + 1, pacmanIndex,
                                               alpha, beta)
            if successorValue < alpha:
                return successorValue
            beta = min(beta, successorValue)
            successorValues.append(successorValue)
        return min(successorValues)

    def Action(self, gameState):
        ####################### Write Your Code Here ################################

        pacmanIndex = 0
        firstGhostIndex = pacmanIndex + 1
        move_candidate = gameState.getLegalActions(pacmanIndex)
        alpha = float("-inf")
        beta = float("inf")
        # value, move=self.maxValue(gameState, 0, pacmanIndex, alpha, beta)

        scores = [self.minValue(gameState.generateSuccessor(pacmanIndex, action), 0, firstGhostIndex, alpha, beta) for
                  action in
                  move_candidate]
        # 팩맨 다음 인덱스부터 minvalue 계산 시작해야
        # print(scores)
        bestScore = max(scores)
        Index = [index for index in range(len(scores)) if scores[index] == bestScore]
        get_index = random.choice(Index)

        return move_candidate[get_index]

        # raise Exception("Not implemented yet")

        ############################################################################


class ExpectimaxAgent(AdversialSearchAgent):
    """
      [문제 03] Expectimax의 Action을 구현하시오. (25점)
      (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
    """

    def maxValue(self, gameState, depth, agentIndex=0):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # terminal of search
            return self.evaluationFunction(gameState)

        move_candidate = gameState.getLegalActions(agentIndex)
        pacmanIndex = 0
        firstGhostIndex = pacmanIndex + 1
        successorChanceValues = [
            self.chanceValue(gameState.generateSuccessor(agentIndex, action), depth, firstGhostIndex)
            for action in move_candidate
        ]
        return max(successorChanceValues)

    def chanceValue(self, gameState, depth, agentIndex=0):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            # terminal of search
            return self.evaluationFunction(gameState)

        move_candidate = gameState.getLegalActions(agentIndex)
        agentNum = gameState.getNumAgents()
        pacmanIndex = 0
        candidateNum = len(move_candidate)
        v = 0

        for action in move_candidate:
            if agentIndex < agentNum - 1:
                v += self.chanceValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                # 다음 인덱스의 chanceValue를 포함시킨다
            else:
                # 다음 depth로 넘기고 팩맨의 움직임으로
                v += self.maxValue(gameState.generateSuccessor(agentIndex, action), depth + 1, pacmanIndex)

        return v / candidateNum

    # 기댓값 리턴

    def Action(self, gameState):
        ####################### Write Your Code Here ################################

        pacmanIndex = 0
        firstGhostIndex = pacmanIndex + 1
        move_candidate = gameState.getLegalActions(pacmanIndex)

        scores = [self.chanceValue(gameState.generateSuccessor(pacmanIndex, action), 0, firstGhostIndex) for action in
                  move_candidate]
        # 팩맨 다음 인덱스부터 minvalue 계산 시작해야
        # print(scores)
        bestScore = max(scores)
        Index = [index for index in range(len(scores)) if scores[index] == bestScore]
        get_index = random.choice(Index)

        return move_candidate[get_index]
        # raise Exception("Not implemented yet")

        ############################################################################
