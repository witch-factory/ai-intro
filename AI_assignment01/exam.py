from collections import deque
import networkx as nx
import networkx.algorithms.approximation


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


def mst(vertex_num, edges):
    cost_sum = 0
    ####################### Write Your Code Here ################################
    parent = [i for i in range(vertex_num)]  # 모든 정점의 루트를 자기 자신으로

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

    edges.sort()

    cnt = 0
    edge_num = len(edges)
    mst_edges = []
    # mst에 들어가는 간선들을 저장한다
    for i in range(edge_num):
        if merge(edges[i].start, edges[i].end):
            mst_edges.append(edges[i])
            cost_sum += edges[i].cost
            cnt += 1
            if cnt == vertex_num - 1:
                break
                # n-1개의 간선이 모이면 mst가 된다
    """for E in mst_edges:
        print(E.start, E.end, E.cost)
    print("")"""
    return mst_edges


class treeNode:
    # 트리의 한 정점을 나타내는 구조체 - preorder를 위해 제작
    def __init__(self, key):
        self.key = key
        self.child = []


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


def mst_euler_path_cost(mst_edges, root):
    cost_sum = 0
    # construct the tree by adjacency list
    vertex_num = 0
    for E in mst_edges:
        vertex_num = max(vertex_num, E.start, E.end)
    vertex_num += 1
    adj_list = [[] for i in range(vertex_num)]

    for E in mst_edges:
        adj_list[E.start].append(E.end)
        adj_list[E.end].append(E.start)

        """for node in queue:
            print(node.key, end=" ")
        print("")"""
    # preorder for the constructed tree
    pre = preorder(root, adj_list)
    return pre

l=[[3,2,1], [2,3,1]]

for it in l:
    it.sort()

for it in l:
    print(it)

"""edges = [
    Edge(0, 1, 1),
    Edge(2, 3, 1),
    Edge(5, 7, 1),
    Edge(6, 9, 1),
    Edge(1, 4, 3),
    Edge(2, 4, 3),
    Edge(3, 6, 3),
    Edge(4, 8, 4),
    Edge(4, 5, 5)
]

mst_edges = mst(10, edges)
for E in mst_edges:
    print(E.start, E.end, E.cost)

print(mst_euler_path_cost(mst_edges, 0))"""
