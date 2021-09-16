# -*- coding: utf-8 -*-
###### Write Your Library Here ###########
import collections
from heapq import *


#########################################


def search(maze, func):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)


# -------------------- Stage 01: One circle - BFS Algorithm ------------------------ #

def bfs(maze):
    """
    [문제 01] 제시된 stage1의 맵 세가지를 BFS Algorithm을 통해 최단 경로를 return하시오.(20점)
    """
    start_point=maze.startPoint()
    #print(maze._Maze__objective)
    path = []
    ####################### Write Your Code Here ################################
    q = collections.deque([start_point])
    prev_visited={start_point:(-1, -1)}
    #이전에 방문했던 점을 저장해 놓으면 경로를 역추적 가능하다
    # 방문한 목적지들을 저장해 놓는다
    while q:
        cur_point=q.pop()
        neighbors=maze.neighborPoints(cur_point[0], cur_point[1])
        # 갈 수 있는 점들을 알아서 뽑아내준다
        for next_point in neighbors:
            if next_point in prev_visited:
                continue
            prev_visited[next_point] = cur_point
            #print(next_point)
            if maze.isObjective(next_point[0], next_point[1]):
                track=next_point
                while track!=(-1,-1):
                    path.append(track)
                    #print(track)
                    track=prev_visited[track]
                break
            q.appendleft(next_point)
        #print(path)
    return path

    ############################################################################



class Node:
    # 휴리스틱 함수를 포함하는 노드이다
    def __init__(self,parent,location):
        self.parent=parent #이전에 방문한 노드, 즉 부모 노드
        self.location=location #현재 노드

        self.obj=[]

        # F = G+H
        self.f=0
        self.g=0
        self.h=0 #휴리스틱 거리

    def __eq__(self, other):
        return self.location==other.location and str(self.obj)==str(other.obj)

    def __le__(self, other):
        return self.g+self.h<=other.g+other.h

    def __lt__(self, other):
        return self.g+self.h<other.g+other.h

    def __gt__(self, other):
        return self.g+self.h>other.g+other.h

    def __ge__(self, other):
        return self.g+self.h>=other.g+other.h


# -------------------- Stage 01: One circle - A* Algorithm ------------------------ #

def manhattan_dist(p1,p2):
    return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

def astar(maze):

    """
    [문제 02] 제시된 stage1의 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.(20점)
    (Heuristic Function은 위에서 정의한 manhattan_dist function을 사용할 것.)
    """

    start_point=maze.startPoint()

    end_point=maze.circlePoints()[0]
    # 목적지이다
    path=[]

    ####################### Write Your Code Here ################################
    priority_queue=[]
    heappush(priority_queue, Node(None, start_point))
    # start 노드부터 시작
    while priority_queue:
        cur=heappop(priority_queue)
        if cur.location in path:continue
        if cur.location==end_point:
            track=cur
            while track.parent is not None:
                path.append(track.location)
                track=track.parent
            break

        neighbors=maze.neighborPoints(cur.location[0], cur.location[1])
        # 갈 수 있는 곳
        for neighbor in neighbors:
            #방문할 수 있는 이웃 정점
            new_node=Node(cur, neighbor)
            # cur를 바로전에 방문했을 것이다
            new_node.g=cur.g+1
            new_node.h=manhattan_dist(neighbor, end_point)
            new_node.f=new_node.g+new_node.h
            heappush(priority_queue, new_node)









    return path

    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #



def stage2_heuristic():
    pass


def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points=maze.circlePoints()
    end_points.sort()

    path=[]

    ####################### Write Your Code Here ################################


















    return path

    ############################################################################



# -------------------- Stage 03: Many circles - A* Algorithm -------------------- #

def mst(objectives, edges):

    cost_sum=0
    ####################### Write Your Code Here ################################













    return cost_sum

    ############################################################################


def stage3_heuristic():
    pass


def astar_many_circles(maze):
    """
    [문제 04] 제시된 stage3의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage3_heuristic function을 직접 정의하여 사용해야 하고, minimum spanning tree
    알고리즘을 활용한 heuristic function이어야 한다.)
    """

    end_points= maze.circlePoints()
    end_points.sort()

    path=[]

    ####################### Write Your Code Here ################################





















    return path

    ############################################################################
