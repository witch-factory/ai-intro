# -*- coding: utf-8 -*-
###### Write Your Library Here ###########
import collections
import copy
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
    end_point=maze.circlePoints()[0]
    path = []
    ####################### Write Your Code Here ################################
    q = collections.deque([start_point])
    prev_visited={start_point:(-1, -1)}
    cur_point=start_point
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
    return path[::-1]
    #뒤집은 경로를 리턴해 줘야 한다.

    ############################################################################



class Node:
    # 휴리스틱 함수를 포함하는 노드이다
    def __init__(self,parent,location):
        self.parent=parent #이전에 방문한 노드, 즉 부모 노드
        self.location=location #현재 노드

        self.obj=[] #어떤 목적지들을 거쳐왔는지. 현재의 방문상태

        # F = G+H
        self.f=0
        self.g=0
        self.h=0 #휴리스틱 거리

    def __eq__(self, other):
        return self.location==other.location and str(self.obj)==str(other.obj)

    def __le__(self, other):
        return self.f<=other.f

    def __lt__(self, other):
        return self.f<other.f

    def __gt__(self, other):
        return self.f>other.f

    def __ge__(self, other):
        return self.f>=other.f


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
    visited=set()
    #방문한 좌표들을 저장한다
    heappush(priority_queue, Node(None, start_point))
    # start 노드부터 시작
    while priority_queue:
        cur=heappop(priority_queue)
        if cur.location in visited:continue
        if cur.location==end_point:
            # 목적지에 도착
            track=cur
            while track is not None:
                path.append(track.location)
                track=track.parent
            path=path[::-1]
            #reverse the list to gain the right path
            #print(path)
            break

        visited.add(cur.location)

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
    print(path)
    return path

    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #



def stage2_heuristic(node, end_points):
    dist=0
    for end_point in end_points:
        # 아직 방문 안 한 정점들까지의 거리 합
        if end_point in node.obj:continue
        dist+=manhattan_dist(end_point, node.location)
    return dist


def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points=maze.circlePoints()
    end_points.sort()

    path=[]

    ####################### Write Your Code Here ################################

    start_point=maze.startPoint()
    pq=[]
    visited=set()
    start_node=Node(None, start_point)
    start_node.h=stage2_heuristic(start_node, end_points)
    start_node.f=start_node.g+start_node.h
    # 시작 노드의 휴리스틱 거리
    heappush(pq, start_node)
    cur_node=start_node

    while set(cur_node.obj)!=set(end_points):
        cur_node = heappop(pq)
        if cur_node.location in visited:
            continue
        if cur_node.location in cur_node.obj:
            continue

        if cur_node.location in end_points:
            cur_node.obj.append(cur_node.location)
            print(cur_node.obj)
            track = cur_node
            temp_path = []
            while track is not None:
                temp_path.append(track.location)
                track = track.parent
            temp_path.reverse()
            path.extend(temp_path)
            visited=set()
            cur_node.parent=None
            pq=[]
            # 지금까지 거쳐온 경로는 저장

            neighbors = maze.neighborPoints(cur_node.location[0], cur_node.location[1])
            # 갈 수 있는 곳
            for neighbor in neighbors:
                # 방문할 수 있는 이웃 정점
                next_node = Node(cur_node, neighbor)
                # cur를 바로전에 방문했을 것이다
                next_node.obj = cur_node.obj
                next_node.g = cur_node.g + 1
                next_node.h = stage2_heuristic(next_node, end_points)
                next_node.f = next_node.g + next_node.h
                heappush(pq, next_node)
            """new_start_node = cur_node
            cur_node.parent = None
            pq = []
            visited = set()
            heappush(pq, new_start_node)"""

            continue

        visited.add(cur_node.location)

        neighbors = maze.neighborPoints(cur_node.location[0], cur_node.location[1])
        # 갈 수 있는 곳
        for neighbor in neighbors:
            # 방문할 수 있는 이웃 정점
            next_node = Node(cur_node, neighbor)
            # cur를 바로전에 방문했을 것이다
            next_node.obj = cur_node.obj
            next_node.g = cur_node.g + 1
            next_node.h = stage2_heuristic(next_node, end_points)
            next_node.f = next_node.g + next_node.h
            heappush(pq, next_node)

    return path


def astar_four_circles_2(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points=maze.circlePoints()
    end_points.sort()

    path=[]

    ####################### Write Your Code Here ################################

    start_point=maze.startPoint()
    #all_path=[[[] for j in range(len(end_points)+1)] for i in range(len(end_points)+1)]
    # [i][j] 는 i번 목적지에서 j번 목적지로 가는 경로를 넣어 놓는다. 0번은 출발지점

    priority_queue = []
    visited = set()
    # 방문한 좌표들을 저장한다
    start_node=Node(None, start_point)
    heappush(priority_queue, start_node)
    cur_node=start_node
    # start 노드부터 시작
    while set(cur_node.obj)!=set(end_points):
        cur_node = heappop(priority_queue)
        if cur_node.location in visited:
            continue

        cur_goal_point=None
        # 이번에 갈 점을 정한다
        for end_point in end_points:
            if end_point not in cur_node.obj and cur_goal_point==None:
                cur_goal_point=end_point
                continue
            if end_point not in cur_node.obj and \
                stage2_heuristic(end_point, cur_node.location)<stage2_heuristic(cur_goal_point, cur_node.location):
                cur_goal_point=end_point

        if cur_node.location==cur_goal_point:
            # 도달 못한 점만 cur_goal_point가 되므로 도달한 점에 또 도달할 일은 없다
            cur_node.obj.append(cur_node.location)
            # 새로 도달한 목적지를 저장해줌. cur_node.location 이 목적지 중 하나
            track=cur_node
            temp_path=[]
            while track is not None:
                temp_path.append(track.location)
                track=track.parent
            temp_path.reverse()
            path.extend(temp_path)
            # 도착한 목적지를 새 노드로 삼아서 출발
            new_start_node=cur_node
            cur_node.parent=None
            priority_queue = []
            visited = set()
            heappush(priority_queue, new_start_node)

            continue

        visited.add(cur_node.location)

        neighbors = maze.neighborPoints(cur_node.location[0], cur_node.location[1])
        # 갈 수 있는 곳
        for neighbor in neighbors:
            # 방문할 수 있는 이웃 정점
            next_node = Node(cur_node, neighbor)
            # cur를 바로전에 방문했을 것이다
            next_node.g = cur_node.g + 1
            next_node.h = manhattan_dist(neighbor, cur_goal_point)
            next_node.f = next_node.g+next_node.h
            next_node.obj=cur_node.obj
            heappush(priority_queue, next_node)
    print(path)
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
