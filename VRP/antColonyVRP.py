import random
from typing import List
from itertools import accumulate
from GraphDataStructure import Graph

"""
CONSTANTS
"""
# Pheromone relative importance
ALPHA = 1
# Relative importance of heuristic factor
BETA = 5
# Pheromone evaporation coefficient (1 - RHO indicates the persistence factor).
RHO = 0.95


def performAntVRP(graph: Graph, pheromoneMatrix: List[List[List[float]]], visitedNodes: List[int],
                  currentNode: List[int], numOfNodes: int, currentVehicle: int) -> int:
    """
    Determines the step each ant will make through the graph
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :param pheromoneMatrix: Matrix containing the pheromone values for each edge on our graph
    :param visitedNodes: The nodes which any of the ants have already visited
    :param currentNode: The node that each vehicle is currently occupying
    :param numOfNodes: The number of nodes in our graph
    :param currentVehicle: The number of vehicles traversing the graph
    :return: The node which the ant should visit next according to our algorithm
    """
    heuristicMatrix = [[round(1 / graph.adjMatrix[i][j], 5) if i != j and j not in visitedNodes else 0
                        for j in range(numOfNodes)] for i in range(numOfNodes)]

    vehiclePheromones = pheromoneMatrix[currentVehicle]

    cumulativeProbabilities = determineProbabilities(vehiclePheromones[currentNode[currentVehicle]],
                                                     heuristicMatrix[currentNode[currentVehicle]])
    return determineStep(cumulativeProbabilities)


def determineProbabilities(pheromoneRow: List[float], heuristicRow: List[float]) -> List[float]:
    """
    Determines the probability of traversing each edge from a given node
    :param pheromoneRow: The pheromone distribution of each path from a given node
    :param heuristicRow: Used to make shorter paths more likely to be taken
    The weighting of our pheromone and heuristic matrices are determined by the defined ALPHA/BETA constants
    :return: The cumulative probability of visiting each node from our current node
    """
    numerator = [pheromoneRow[i] ** ALPHA * heuristicRow[i] ** BETA for i in range(len(heuristicRow))]
    denominator = sum(numerator)
    probabilities = [num / denominator for num in numerator]
    return list(accumulate(probabilities))


def determineStep(probabilities: List[float]) -> int:
    """
    Generates a random number between [0, 1] to determine which edge the ant will traverse
    :param probabilities: List of cumulative probabilities ranging from [0, 1]
    :return: The edge which should be taken by the ant
    """
    randomStep = random.random()
    for probability in probabilities:
        if randomStep <= probability:
            return probabilities.index(probability)


def evaporatePheromones(pheromoneMatrix: List[List[List[float]]], numberOfNodes: int,
                        numberOfVehicles: int) -> List[List[List[float]]]:
    """
    Evaporates pheromones off each edge using our defined constant RHO
    :param numberOfVehicles: The number of vehicles traversing the graph
    :param pheromoneMatrix: The pheromone distribution of each edge of our graph
    :param numberOfNodes: The number of nodes in our graph
    :return: The pheromone distribution of each edge after evaporation
    """
    pheromones = []
    for i in range(numberOfVehicles):
        pheromones.append([[(1 - RHO) * pheromoneMatrix[i][j][k]
                            for j in range(numberOfNodes)] for k in range(numberOfNodes)])
    return pheromones


def updatePheromones(pheromoneMatrix: List[List[List[float]]], paths: List[List[int]],
                     pathLengths: List[int]) -> List[List[List[float]]]:
    """
    Once a path has been traversed, the ant will deposit pheromones on every edge taken
    :param pheromoneMatrix: The pheromone distribution of each edge of our graph
    :param paths: The path taken by the ant
    :param pathLengths: The length of the path taken by the ant
    :return: The pheromone distribution of each edge after depositing pheromones
    """
    maxPathLength = max(pathLengths)
    deposit = 1 / maxPathLength
    for vehicleIndex in range(len(paths)):
        for i in range(len(paths[vehicleIndex]) - 1):
            pheromoneMatrix[vehicleIndex][paths[vehicleIndex][i]][paths[vehicleIndex][i + 1]] += deposit
    return pheromoneMatrix


def generateVehicleTravelList(n, x):
    """
    Generates and randomises a list indicating the position in which each vehicle travels
    :param n: The number of nodes
    :param x: The number of vehicles
    :return: A list indicating the order all vehicles will travel
    """
    quotient, remainder = divmod(n, x)
    vehicleList = []
    for i in range(x):
        vehicleList += [i]*quotient
        vehicleList += [i]*(remainder > i)
    random.shuffle(vehicleList)
    return vehicleList


def runAntColonyVRP(graph: Graph, startingNode: int = 0, numberOfVehicles: int = 3, numberOfIterations: int = 100):
    """
    Runs the Ant Colony Optimization VRP algorithm on our graph
    :param numberOfVehicles: The number of vehicles which will be traversing the graph
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :param startingNode: Our starting (and ending) node
    :param numberOfIterations: The number of iterations of our algorithm we will perform
    :return: A tuple containing the path generated by the algorithm, and the length of this path
    """
    numOfNodes = graph.size
    pheromoneMatrix = [[[1 for _ in range(numOfNodes)] for _ in range(numOfNodes)] for _ in range(numberOfVehicles)]
    paths = [[]]
    pathLengths = []

    # According to M. Dorigo, the optimal number of ants is 2n/5 where n is number of nodes
    numberOfAnts = max((numOfNodes // 5) * 2, 1)

    for _ in range(numberOfIterations):
        pheromoneMatrix = evaporatePheromones(pheromoneMatrix, numOfNodes, numberOfVehicles)

        for _ in range(numberOfAnts):
            pathLengths = [0 for _ in range(numberOfVehicles)]
            paths = [[startingNode] for _ in range(numberOfVehicles)]
            currentNode = [startingNode for _ in range(numberOfVehicles)]
            visitedNodes = [startingNode]

            vehicleList = generateVehicleTravelList(numOfNodes - 1, numberOfVehicles)
            # Generates a Hamiltonian Cycle for the current ant.
            while len(visitedNodes) != numOfNodes:

                # Randomly select which of the vehicles we should move next.
                currentVehicle = vehicleList[len(visitedNodes) - 1]
                visitingNode = performAntVRP(graph, pheromoneMatrix, visitedNodes,
                                             currentNode, numOfNodes, currentVehicle)
                paths[currentVehicle].append(visitingNode)
                pathLengths[currentVehicle] += graph.adjMatrix[currentNode[currentVehicle]][visitingNode]
                visitedNodes.append(visitingNode)
                currentNode[currentVehicle] = visitingNode

            for vehicle in range(numberOfVehicles):
                paths[vehicle].append(startingNode)
                pathLengths[vehicle] += graph.adjMatrix[currentNode[vehicle]][startingNode]
            # Deposits pheromones onto the path that the ant took.
            pheromoneMatrix = updatePheromones(pheromoneMatrix, paths, pathLengths)
    return paths, max(pathLengths)
