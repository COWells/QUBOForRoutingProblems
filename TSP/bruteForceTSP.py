import numpy as np
from typing import List, Tuple, Generator
from itertools import permutations
from GraphDataStructure import Graph


def createFullRoute(iterator: Generator, startNode: Tuple[int]) -> List[Tuple[int]]:
    """
    Appends the starting node to the start and end of each route
    :param iterator: Contains all permutations of possible routes
    :param startNode: The node at which we must start and finish
    :return: The full route with the starting and ending node appended (1, 2, 3) -> (0, 1, 2, 3, 0)
    """
    for route in iterator:
        yield list(startNode + route + startNode)


def generateRoutes(nodesToVisit: List[int]):
    """
    Generate every valid, non-repeated route, not including the starting node
    :param nodesToVisit: List of each node which should be visited before returning to the start node
    :return: A reduced list of each route
    """
    for permutation in permutations(nodesToVisit):
        if permutation[0] <= permutation[-1]:
            yield permutation


def runBruteForceTSP(graph: Graph, startingNode: int = 0) -> Tuple[Tuple[int], int]:
    """
    Runs the brute force TSP algorithm on our graph
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :param startingNode: Our starting (and ending) node
    :return: A tuple containing the minimum path and the length of this path
    """
    nodesToVisit = [x for x in range(graph.size) if x != startingNode]
    minPath = None
    minPathLength = np.inf

    # Find the route with the minimum total distance (Minimum Hamiltonian Cycle).
    for route in createFullRoute(generateRoutes(nodesToVisit), (startingNode,)):
        totalDist = sum(graph.adjMatrix[route[x]][route[x + 1]] for x in range(0, len(route) - 1))

        if totalDist < minPathLength:
            minPathLength = totalDist
            minPath = route

    return minPath, minPathLength
