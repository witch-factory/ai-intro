# -*- coding: utf-8 -*-
# --------- Get the Maze Information --------- ##

import re
import copy


class Maze:

    def __init__(self, filename):
        self.__filename = filename
        self.__wall = '#'
        self.__startchar = 'T'
        self.__objectivechar = '.'
        self.__start = None
        self.__objective = []
        self.__states_search = 0

        with open(filename) as f:
            lines = f.readlines()
        # 파일 읽어오기
        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        lines = [list(line.strip('\n')) for line in lines]
        # print(lines)
        self.rows = len(lines)
        self.cols = len(lines[0])
        self.mazeRaw = lines

        if (len(self.mazeRaw) != self.rows) or (len(self.mazeRaw[0]) != self.cols):
            # 미로의 dimension 체크
            print("Maze dimensions incorrect")
            raise SystemExit
            return

        for row in range(len(self.mazeRaw)):
            for col in range(len(self.mazeRaw[0])):
                if self.mazeRaw[row][col] == self.__startchar:
                    # 나는 start(T로 표시)에서 시작한다
                    self.__start = (row, col)
                elif self.mazeRaw[row][col] == self.__objectivechar:
                    # '.' 으로 표시되는 곳이 목적지
                    self.__objective.append((row, col))

    def isWall(self, row, col):
        return self.mazeRaw[row][col] == self.__wall

    # 벽인지 체크

    def isObjective(self, row, col):
        return (row, col) in self.__objective

    # 목적지인지 체크

    ## 시작 위치의 tuple(row, col)을 return
    def startPoint(self):
        return self.__start

    # start 점을 새로 세팅
    def setStart(self, start):
        self.__start = start

    def getDimensions(self):
        return (self.rows, self.cols)

    ## 원 위치에 해당하는 tuple list[(row1, col1), (row2, col2)]을 return
    def circlePoints(self):
        return copy.deepcopy(self.__objective)

    # 목적지들을 인수로 줘서 그걸 목적지 리스트로 세팅
    def setObjectives(self, objectives):
        self.__objective = objectives

    def getStatesSearch(self):
        return self.__states_search

    #그곳으로 갈 수 있는지, 즉 선택한 (r,c) 점이 영역을 벗어나지않으며 벽이 아닌지
    def choose_move(self, row, col):
        return row >= 0 and row < self.rows and col >= 0 and col < self.cols and not self.isWall(row, col)

    ## 현재 위치에서 인접 위치에 해당하는 tuple list를 return
    # 단 이동할 수 있는 위치들만 리턴
    def neighborPoints(self, row, col):
        possibleNeighbors = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1)
        ]
        neighbors = []
        for r, c in possibleNeighbors:
            if self.choose_move(r, c):
                neighbors.append((r, c))
        self.__states_search += 1
        return neighbors

    # 유효한 이동 경로인지 체크
    def isValidPath(self, path):
        for i in range(1, len(path)):
            prev = path[i - 1]
            cur = path[i]
            dist = abs((prev[1] - cur[1]) + (prev[0] - cur[0]))
            if dist > 1:
                return "Not single hop"
            # 한칸씩 이동한 게 아님

        for pos in path:
            if not self.choose_move(pos[0], pos[1]):
                return "Not valid move"
            # 갈 수 있는 지점이 아닌데 경로에 있음

        if not set(self.__objective).issubset(set(path)):
            return "Not all goals passed"
        # 모든 목적지에 도달하지 않았음

        if not path[-1] in self.__objective:
            return "Last position is not goal"
        # 마지막에 목적지에 있지 않음
        return "Valid"
    # 이 모든 조건을 피해갔다면 유효한 경로임
