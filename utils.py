import time
from typing import Tuple, List, Callable
import numpy as np
from GraphDataStructure import Graph


def formatPath(path: Tuple[int]) -> str:
    """
    Formats the path into a readable form to display to user
    :param path: The path (Hamiltonian Cycle) taken
    :return: Formats the path to show how it was traversed. [0, 1, 2, 0] becomes "0 -> 1 -> 2 -> 0"
    """
    return "->".join(f"{x}" for x in path)


def formatVRPPath(VRPPath: List[Tuple[int]]):
    """
    Formats a list of paths into a readable form to display to user
    :param VRPPath: The paths taken
    :return: Formats the path to show how it was traversed. [0, 1, 2, 0] becomes "0 -> 1 -> 2 -> 0"
    """
    return ["Vehicle {num}: ".format(num=i + 1) + formatPath(VRPPath[i]) for i in range(len(VRPPath))]


def printPath(algorithmName: str, path: str, pathLength: int):
    """
    Prints the results to the console
    :param algorithmName: The name of the algorithm which has been run
    :param path: The path (Hamiltonian Cycle) taken
    :param pathLength: The length of the path
    """
    print(algorithmName, "Path:", path)
    print(algorithmName, "Distance:",  pathLength)


def printVRPPath(algorithmName: str, path: List[str], pathLength: int):
    """
    Prints the results to the console
    :param algorithmName: The name of the algorithm which has been run
    :param path: The paths taken
    :param pathLength: The length of the path
    """
    print(algorithmName, "VRP Path:")
    for vehicleRoute in path:
        print(vehicleRoute)
    print(algorithmName, "Distance:", pathLength)


def generateRandomGraph(size: int = 10) -> Graph:
    """
    Generates a random, valid graph of the given size
    :param size: The number of cities which should be included in the network
    :return: An instance of the graph data structure with a random adjacency matrix of the correct size
    """
    adjMatrix = np.random.randint(1, 100, size=(size, size))
    for i in range(len(adjMatrix)):
        adjMatrix[i][i] = 0
    graph = Graph(size)
    graph.addMatrix((adjMatrix + adjMatrix.T) // 2)
    return graph


def determineExecutionTime(routeAlgorithm: Callable, graph: Graph, numberOfVehicles: int = 1):
    """
    Determines the runtime for a given routing algorithm and graph
    :param routeAlgorithm: The algorithm which will be used
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :param numberOfVehicles: The number of vehicles that will be traversing the network
    """
    pre = time.process_time()
    routeAlgorithm(graph, numberOfVehicles) if numberOfVehicles > 1 else routeAlgorithm(graph)
    post = time.process_time()
    print('CPU Execution time', post - pre, 'seconds')
    print("===========================================")
