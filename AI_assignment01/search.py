# -*- coding: utf-8 -*-
###### Write Your Library Here ###########
from collections import deque
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
    start_point = maze.startPoint()
    # print(maze._Maze__objective)
    end_point = maze.circlePoints()[0]
    path = []
    ####################### Write Your Code Here ################################
    q = deque([start_point])
    prev_visited = {start_point: (-1, -1)}
    cur_point = start_point
    # 이전에 방문했던 점을 저장해 놓으면 경로를 역추적 가능하다
    # 방문한 목적지들을 저장해 놓는다
    while q:
        cur_point = q.pop()
        neighbors = maze.neighborPoints(cur_point[0], cur_point[1])
        # 갈 수 있는 점들을 알아서 뽑아내준다
        for next_point in neighbors:
            if next_point in prev_visited:
                continue
            prev_visited[next_point] = cur_point
            # print(next_point)
            if maze.isObjective(next_point[0], next_point[1]):
                track = next_point
                while track != (-1, -1):
                    path.append(track)
                    # print(track)
                    track = prev_visited[track]
                break
            q.appendleft(next_point)
    return path[::-1]
    # 뒤집은 경로를 리턴해 줘야 한다.

    ############################################################################


class Node:
    # 휴리스틱 함수를 포함하는 노드이다
    def __init__(self, parent, location):
        self.parent = parent  # 이전에 방문한 노드, 즉 부모 노드
        self.location = location  # 현재 노드

        self.obj = []  # 어떤 목적지들을 거쳐왔는지. 현재의 방문상태

        # F = G+H
        self.f = 0
        self.g = 0
        self.h = 0  # 휴리스틱 거리

    def __eq__(self, other):
        return self.location == other.location and str(self.obj) == str(other.obj)

    def __le__(self, other):
        return self.f <= other.f

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f


# -------------------- Stage 01: One circle - A* Algorithm ------------------------ #

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def astar(maze):
    """
    [문제 02] 제시된 stage1의 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.(20점)
    (Heuristic Function은 위에서 정의한 manhattan_dist function을 사용할 것.)
    """

    start_point = maze.startPoint()

    end_point = maze.circlePoints()[0]
    # 목적지이다
    path = []

    ####################### Write Your Code Here ################################
    priority_queue = []
    visited = set()
    # 방문한 좌표들을 저장한다
    heappush(priority_queue, Node(None, start_point))
    # start 노드부터 시작
    while priority_queue:
        cur = heappop(priority_queue)
        if cur.location in visited: continue
        if cur.location == end_point:
            # 목적지에 도착
            track = cur
            while track is not None:
                path.append(track.location)
                track = track.parent
            path = path[::-1]
            # reverse the list to gain the right path
            # print(path)
            break

        visited.add(cur.location)

        neighbors = maze.neighborPoints(cur.location[0], cur.location[1])
        # 갈 수 있는 곳
        for neighbor in neighbors:
            # 방문할 수 있는 이웃 정점
            new_node = Node(cur, neighbor)
            # cur를 바로전에 방문했을 것이다
            new_node.g = cur.g + 1
            new_node.h = manhattan_dist(neighbor, end_point)
            new_node.f = new_node.g + new_node.h
            heappush(priority_queue, new_node)
    return path

    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #


def stage2_heuristic(node, end_points, dist_between_goals):
    dist = 0

    candidate_dists = []
    for candidate in end_points:
        # 모든 정점을 한번씩 방문해 보고, 그 순서 중에 최소 비용
        # 일단 제일 먼저 방문할 정점을 잡고 나면 그리디하게 접근함
        visited_goals = set(node.obj)
        visited_goals.add(candidate)
        # 모든 goal을 다 방문할 때까지의 거리를 구해야 한다
        cur_goal = candidate
        all_visit_cost = 0
        # 아직 안 방문한 모든 정점을 방문하는 데에 드는 비용
        while visited_goals != set(end_points):
            # 가장 가까운 목표에서, 아직 안 방문한 목표들을 모두 방문하는 데 얼만큼의 비용이 드는지
            next_goal = None
            for end_point in end_points:
                # 더 적은 비용으로 갈 수 있는, 아직 안 방문한 정점을 찾는다
                if end_point in visited_goals: continue
                # if end_point==nearest_goal:continue
                if next_goal == None:
                    next_goal = end_point
                else:
                    if dist_between_goals[end_points.index(cur_goal) + 1][end_points.index(end_point) + 1] < \
                            dist_between_goals[end_points.index(cur_goal) + 1][end_points.index(next_goal) + 1]:
                        # 더 적은 비용으로 갈 수 있는 목표를 바로 다음에 간다
                        next_goal = end_point
            visited_goals.add(next_goal)
            # 새로 방문한 목표
            all_visit_cost += dist_between_goals[end_points.index(cur_goal) + 1][end_points.index(next_goal) + 1]
            cur_goal = next_goal
        candidate_dist = all_visit_cost + manhattan_dist(candidate, node.location)
        # 그 candidate에서 다른 모든 목표를 방문해 주는 데에 얼마나 걸리는지 + 현위치에서 그 candidate까지 얼마나 걸리는지의 맨해튼거리
        candidate_dists.append(candidate_dist)
    dist = min(candidate_dists)

    return dist


def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points = maze.circlePoints()
    end_points.sort()

    path = []

    ####################### Write Your Code Here ################################
    start_point = maze.startPoint()
    path.append(start_point)
    all_goal_dist = [[0 for j in range(len(end_points) + 1)] for i in range(len(end_points) + 1)]
    # 모든 노드들 간의 거리를 전처리로 구해 놓는다. node[i][j] 는 i번, j번 goal사이의 거리
    # 휴리스틱 함수에서 사용하며, 나중에 모든 goal들 각각 간의 경로도 구해 놓는 식으로 전처리 최적화 가능
    # 일단은 거리만 구해놓고 경로는 나중에 온라인으로 구하자
    # print(end_points)
    for goal_idx in range(len(end_points)):
        goal = end_points[goal_idx]
        bfs_start_node = Node(None, goal)
        q = deque([bfs_start_node])
        # 각 goal에서 시작한다
        # prev_visited = {goal: (-1, -1)}
        # cur_point = goal
        # 이전에 방문했던 점을 저장해 놓으면 경로를 역추적 가능하다
        # 방문한 목적지들을 저장해 놓는다
        visited_goal = set()
        left_goal = set()
        visited = set()
        visited.add(goal)
        for left in end_points:
            if left != goal:
                # 우리가 시작한 goal 외에는 모두 방문해야 할 점들
                left_goal.add(left)
        while q and left_goal != visited_goal:
            cur_node = q.pop()
            cur_point = cur_node.location
            neighbors = maze.neighborPoints(cur_point[0], cur_point[1])
            # 갈 수 있는 점들을 알아서 뽑아내준다
            for next_point in neighbors:
                if next_point in visited:
                    continue
                visited.add(next_point)
                # print(next_point)
                if maze.isObjective(next_point[0], next_point[1]):
                    # path_between_goal=[]
                    visited_goal.add(next_point)
                    # visited=set()
                    track = next_point
                    """while track != (-1, -1):
                        path_between_goal.append(track)
                        # print(track)
                        track = prev_visited[track]"""
                    all_goal_dist[end_points.index(next_point) + 1][goal_idx + 1] = cur_node.g + 1
                    all_goal_dist[goal_idx + 1][end_points.index(next_point) + 1] = cur_node.g + 1
                next_node = Node(cur_node, next_point)
                next_node.g = cur_node.g + 1
                q.appendleft(next_node)
    # print(all_goal_dist)

    pq = []
    visited = set()
    start_node = Node(None, start_point)
    start_node.h = stage2_heuristic(start_node, end_points, all_goal_dist)
    start_node.f = start_node.g + start_node.h
    # 시작 노드의 휴리스틱 거리
    heappush(pq, start_node)
    cur_node = start_node

    # 거리 디버깅
    """for end_point in end_points:
        goal_node=Node(None, end_point)
        print(end_point, stage2_heuristic(goal_node, end_points, all_goal_dist))"""

    while set(cur_node.obj) != set(end_points) and pq:
        cur_node = heappop(pq)
        if cur_node.location in visited:
            continue
        if cur_node.location in cur_node.obj:
            continue

        if cur_node.location in end_points and cur_node.location not in cur_node.obj:
            # 목표에 도달했으며, 아직 방문하지 않은 목표일 경우
            # 이미 방문한 목표인 경우에는 신경쓰지 않고 지나가야 함
            track = cur_node
            temp_path = []
            while track.parent is not None:
                temp_path.append(track.location)
                track = track.parent
            temp_path.reverse()
            path.extend(temp_path)
            visited = set()
            cur_node.parent = None
            pq = []
            # 지금까지 거쳐온 경로는 저장
            # pq 초기화. 만약 중복해서 지나야 하는 goal 이 있다면 pq 초기화를 안해야함.
            # stage3을 이걸로 할거면 pq=[]부분 주석처리
            cur_node.obj.append(cur_node.location)
            # print(cur_node.obj)
            if set(cur_node.obj) == set(end_points):
                # 모든 목표를 방문했으면 끝내야 한다
                break

            neighbors = maze.neighborPoints(cur_node.location[0], cur_node.location[1])
            # 갈 수 있는 곳
            for neighbor in neighbors:
                # 방문할 수 있는 이웃 정점
                next_node = Node(cur_node, neighbor)
                # cur를 바로전에 방문했을 것이다
                next_node.obj = cur_node.obj
                next_node.g = cur_node.g + 1
                next_node.h = stage2_heuristic(next_node, end_points, all_goal_dist)
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
            next_node.h = stage2_heuristic(next_node, end_points, all_goal_dist)
            next_node.f = next_node.g + next_node.h
            heappush(pq, next_node)
            # print(next_node.location, next_node.h)
    print(path)
    return path

    ############################################################################


# -------------------- Stage 03: Many circles - A* Algorithm -------------------- #

class Edge:
    # 크루스칼 알고리즘을 위한 간선 구조체
    def __init__(self, start, end, cost):
        self.start = start
        self.end = end
        self.cost = cost

    # 비교 연산자 오버로딩
    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost


def mst(cur_node, end_points, all_goal_dist):
    cost_sum = 0
    ####################### Write Your Code Here ################################
    parent = [i for i in range(len(end_points))]  # 모든 정점의 루트를 자기 자신으로

    def Find(a):
        if a == parent[a]:
            return a
        else:
            t = Find(parent[a])
            parent[a] = t
            return t

    def merge(a, b):
        a = Find(a)
        b = Find(b)
        if a == b: return 0
        # 합병 실패
        parent[b] = a
        return 1

    # merge 성공시 1

    unvisited_goal_num = len(end_points) - len(cur_node.obj)
    goal_num = len(end_points)
    # 아직 방문 안한 목표들 개수
    edges = []
    for i in range(goal_num):
        # 방문하지 않은 정점에 대한 edge들을 모두 저장한다
        if end_points[i] in cur_node.obj: continue
        for j in range(goal_num):
            if i == j: continue
            if end_points[j] in cur_node.obj: continue
            # 이미 방문한 정점에 대한 간선이면 mst에 넣을 필요가 없다
            edges.append(Edge(i, j, all_goal_dist[i][j]))
            # end_point의 인덱스는 0부터

    edges.sort()

    cnt = 0
    edge_num = len(edges)
    mst_edges = []
    # mst에 들어가는 간선들을 저장한다
    for i in range(edge_num):
        s=edges[i].start
        e=edges[i].end
        if s==Find(s) and e==Find(e) and merge(s, e):
            # 끝끼리 병합할 때만 합쳐준다
            mst_edges.append(edges[i])
            cost_sum += edges[i].cost
            cnt += 1
            if cnt == unvisited_goal_num - 1:
                break
                # n-1개의 간선이 모이면 mst가 된다
    """for E in mst_edges:
        print(E.start, E.end, E.cost)
    print("")"""
    return mst_edges


def preorder(root, adj_list):
    # preorder make same result to dfs in tree
    nodes = deque([root])
    # dfs by stack
    order = []
    visited = set()
    visited.add(root)
    while nodes:
        cur = nodes.pop()
        order.append(cur)
        for next in adj_list[cur]:
            if next in visited: continue
            visited.add(next)
            nodes.append(next)
    return order


def mst_euler_path_cost(mst_edges, root, all_goal_dist):
    cost_sum = 0
    # construct the tree by adjacency list

    vertex_num = len(all_goal_dist)

    adj_list = [[] for i in range(vertex_num)]

    for E in mst_edges:
        adj_list[E.start].append(E.end)
        adj_list[E.end].append(E.start)

    # print(adj_list)
    pre = preorder(root, adj_list)
    # print(pre)
    preorder_length = len(pre)
    for i in range(1, preorder_length):
        cost_sum += all_goal_dist[pre[i - 1]][pre[i]]
    # preorder순서대로 따라가면서 거리를 더해준다
    return cost_sum


############################################################################


def stage3_heuristic(cur_node, end_points, all_goal_dist):
    unvisited_goals = [end_point for end_point in end_points if end_point not in cur_node.obj]

    nearest_goal = None
    for goal in unvisited_goals:
        if nearest_goal == None:
            nearest_goal = goal
        else:
            if manhattan_dist(goal, cur_node.location) < manhattan_dist(nearest_goal, cur_node.location):
                nearest_goal = goal

    dist_to_goal = manhattan_dist(nearest_goal, cur_node.location)
    # 현재 위치에서 그 목표까지의 맨해튼 거리
    mst_edges = mst(cur_node, end_points, all_goal_dist)
    mst_cost = mst_euler_path_cost(mst_edges, end_points.index(nearest_goal), all_goal_dist)
    dist = dist_to_goal + mst_cost
    return dist


def astar_many_circles(maze):
    """
    [문제 04] 제시된 stage3의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage3_heuristic function을 직접 정의하여 사용해야 하고, minimum spanning tree
    알고리즘을 활용한 heuristic function이어야 한다.)
    """

    end_points = maze.circlePoints()
    end_points.sort()
    path = []

    ####################### Write Your Code Here ################################
    start_point = maze.startPoint()
    path.append(start_point)
    all_goal_dist = [[0 for j in range(len(end_points))] for i in range(len(end_points))]
    # 모든 노드들 간의 거리를 전처리로 구해 놓는다. node[i][j] 는 i번, j번 goal사이의 거리
    end_len=len(end_points)
    for goal_idx in range(end_len):

        # 모든 목표 노드들 간의 거리 구하기
        goal = end_points[goal_idx]
        bfs_start_node = Node(None, goal)
        q = deque([bfs_start_node])
        # 각 goal에서 시작한다
        visited_goal = set()
        left_goal = set()
        visited = set()
        visited.add(goal)
        for left in end_points:
            if left != goal:
                # 우리가 시작한 goal 외에는 모두 방문해야 할 점들
                left_goal.add(left)
        while q and left_goal != visited_goal:
            cur_node = q.pop()
            cur_point = cur_node.location
            neighbors = maze.neighborPoints(cur_point[0], cur_point[1])
            # 갈 수 있는 점들을 알아서 뽑아내준다
            for next_point in neighbors:
                if next_point in visited:
                    continue
                visited.add(next_point)
                # print(next_point)
                if maze.isObjective(next_point[0], next_point[1]):
                    # path_between_goal=[]
                    visited_goal.add(next_point)
                    # visited=set()
                    track = next_point
                    """while track != (-1, -1):
                        path_between_goal.append(track)
                        # print(track)
                        track = prev_visited[track]"""
                    all_goal_dist[end_points.index(next_point)][goal_idx] = cur_node.g + 1
                    all_goal_dist[goal_idx][end_points.index(next_point)] = cur_node.g + 1
                next_node = Node(cur_node, next_point)
                next_node.g = cur_node.g + 1
                q.appendleft(next_node)

    #print(all_goal_dist)

    edges = []
    for i in range(len(end_points)):
        for j in range(len(end_points)):
            if i == j: continue
            edges.append(Edge(i, j, all_goal_dist[i][j]))
    # 간선들을 모두 뽑아서 리스트 만들기 - mst 제작을 위해서
    edges.sort()

    """mst_edges=mst(cur_node, end_points, all_goal_dist)
    for E in mst_edges:
        print(E.start, E.end, E.cost)"""
    # print(stage3_heuristic(cur_node, end_points, all_goal_dist))

    pq = []
    visited = set()
    start_node = Node(None, start_point)
    start_node.h = stage3_heuristic(start_node, end_points, all_goal_dist)
    start_node.f = start_node.g + start_node.h
    # 시작 노드의 휴리스틱 거리
    heappush(pq, start_node)
    cur_node = start_node

    # 거리 디버깅
    """for end_point in end_points:
        goal_node=Node(None, end_point)
        print(end_point, stage2_heuristic(goal_node, end_points, all_goal_dist))"""

    while set(cur_node.obj) != set(end_points) and pq:
        # print(visited)
        cur_node = heappop(pq)
        if cur_node.location in visited:
            # 이미 방문한 노드일 경우 스킵
            continue

        if cur_node.location in end_points and cur_node.location not in cur_node.obj:
            # 목표에 도달했으며, 아직 방문하지 않은 목표일 경우 경로를 저장해 준다
            # 이미 방문한 목표인 경우에는 신경쓰지 않고 지나가야 함
            track = cur_node
            temp_path = []
            while track.parent is not None:
                temp_path.append(track.location)
                track = track.parent
            temp_path.reverse()
            path.extend(temp_path)
            visited = set()
            cur_node.parent = None
            pq = []
            # 지금까지 거쳐온 경로는 저장
            # pq 초기화. 만약 중복해서 지나야 하는 goal 이 있다면 pq 초기화를 안해야함.
            # stage3을 이걸로 할거면 pq=[]부분 주석처리
            cur_node.obj.append(cur_node.location)
            # print(cur_node.obj)
            if set(cur_node.obj) == set(end_points):
                # 모든 목표를 방문했으면 끝내야 한다
                break

            neighbors = maze.neighborPoints(cur_node.location[0], cur_node.location[1])
            # 갈 수 있는 곳
            for neighbor in neighbors:
                # 방문할 수 있는 이웃 정점
                next_node = Node(cur_node, neighbor)
                # cur를 바로전에 방문했을 것이다
                next_node.obj = cur_node.obj
                next_node.g = cur_node.g + 1
                next_node.h = stage3_heuristic(next_node, end_points, all_goal_dist)
                next_node.f = next_node.g + next_node.h
                # print(next_node.h)
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
            next_node.h = stage3_heuristic(next_node, end_points, all_goal_dist)
            next_node.f = next_node.g + next_node.h
            heappush(pq, next_node)
            # print(next_node.location, next_node.h)
    print(path)
    print(maze.isValidPath(path))

    return path

    ############################################################################
