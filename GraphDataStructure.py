from typing import List


class Graph(object):
    """
    Class used to represent our graph data structure
    adjMatrix: Adjacency matrix storing the distances between each vertex
    size: Size of the adjacency matrix
    """

    def __init__(self, size: int):
        """
        Creates an adjacency matrix with all edges being set to a weight of 0
        :param size: The size of the adjacency matrix
        """
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for _ in range(size)])
        self.size = size

    def addEdge(self, v1: int, v2: int, value: int):
        """
        Sets the (bidirectional) edge between two vertices to a specific value
        :param v1: Vertex 1
        :param v2: Vertex 2
        :param value: The weight of the edge between vertex 1 and vertex 2
        """
        if v1 == v2 and value != 0:
            print("Same vertex %d and %d" % (v1, v2))
            value = 0
        self.adjMatrix[v1][v2] = value
        self.adjMatrix[v2][v1] = value

    def addRow(self, row: int, values: List[int]):
        """
        Sets a specific row of the adjacency matrix to a given list
        :param row: The row number which we wish to change
        :param values: The values of the row
        """
        self.adjMatrix[row] = values

    def addMatrix(self, matrix: List[List[int]]):
        """
        Sets the adjacency matrix to a given 2d array
        :param matrix: A 2d array which we will set as the adjacency matrix
        """
        self.adjMatrix = matrix