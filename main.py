import string
import sys
from typing import List, Tuple

import numpy as np
import TSP
import VRP
import customExceptions
from GraphDataStructure import Graph
from utils import formatPath, formatVRPPath, printPath, printVRPPath, generateRandomGraph, determineExecutionTime


def runBruteForceTSP(graph):
    (bruteForcePathTSP, bruteForcePathLengthTSP) = TSP.runBruteForceTSP(graph=graph, startingNode=0)
    printPath("Brute Force", formatPath(bruteForcePathTSP), bruteForcePathLengthTSP)
    return bruteForcePathLengthTSP


def runNearestNeighbourTSP(graph):
    (nearestNeighbourPathTSP, nearestNeighbourPathLengthTSP) = TSP.runNearestNeighbourTSP(graph=graph,
                                                                                          startingNode=0)
    printPath("Nearest Neighbour", formatPath(nearestNeighbourPathTSP), nearestNeighbourPathLengthTSP)
    return nearestNeighbourPathLengthTSP


def runAntColonyTSP(graph):
    (antColonyPathTSP, antColonyPathLengthTSP) = TSP.runAntColonyTSP(graph=graph, startingNode=0,
                                                                     numberOfIterations=100)
    printPath("Ant Colony", formatPath(antColonyPathTSP), antColonyPathLengthTSP)
    return antColonyPathLengthTSP


def runQUBOTSP(graph):
    (QUBOPathTSP, QUBOPathLengthTSP) = TSP.runQuboTSP(graph=graph)
    printPath("QUBO", formatPath(QUBOPathTSP), QUBOPathLengthTSP)
    return QUBOPathLengthTSP


def runBruteForceVRP(graph, numberOfVehicles):
    (bruteForcePathVRP, bruteForcePathLengthVRP) = VRP.runBruteForceVRP(graph=graph, startingNode=0,
                                                                        numberOfVehicles=numberOfVehicles)
    printVRPPath("Brute Force", formatVRPPath(bruteForcePathVRP), bruteForcePathLengthVRP)
    return bruteForcePathLengthVRP


def runNearestNeighbourVRP(graph, numberOfVehicles):
    (nearestNeighbourPathVRP, nearestNeighbourPathLengthVRP) = VRP.runNearestNeighbourVRP(graph, startingNode=0,
                                                                                          numberOfVehicles=numberOfVehicles)
    printVRPPath("Nearest Neighbour", formatVRPPath(nearestNeighbourPathVRP), nearestNeighbourPathLengthVRP)
    return nearestNeighbourPathLengthVRP


def runAntColonyVRP(graph, numberOfVehicles):
    antColonyPathVRP, antColonyPathLengthVRP = VRP.runAntColonyVRP(graph, startingNode=0,
                                                                   numberOfVehicles=numberOfVehicles)
    printVRPPath("Ant Colony", formatVRPPath(antColonyPathVRP), antColonyPathLengthVRP)
    return antColonyPathLengthVRP


def runQUBOVRP(graph, numberOfVehicles):
    quboPathVRP, quboPathLengthVRP = VRP.runQUBOVRP(graph, startingNode=0, numberOfVehicles=numberOfVehicles)
    printVRPPath("QUBO", formatVRPPath(quboPathVRP), quboPathLengthVRP)
    return quboPathLengthVRP


def verifyValidGraph(adjMatrix: np.ndarray):
    """
    Verify that the graph supplied in the text file is valid (Symmetric, 0 on leading diagonal, non-0 everywhere else)
    :param adjMatrix: Numpy array which was parsed in from the supplied text file
    """

    # Verify that the matrix is square (same number of rows and columns, all of which have the same length).
    if not all(len(row) == len(adjMatrix) for row in adjMatrix):
        print("The supplied matrix is not square, please ensure every row is the same size, and that the number "
              "of rows is the same as the number of columns")
        raise customExceptions.MatrixNotSquareException

    # Verify that the matrix is symmetric (It is exactly equivalent to its transpose).
    if np.any(adjMatrix != adjMatrix.T):
        print("Error: Please ensure that the supplied matrix is symmetric")
        raise customExceptions.MatrixNotSymmetricException

    # Verify that the matrix has 0's on the leading diagonal, and non-zeros everywhere else.
    for i in range(0, len(adjMatrix)):
        for j in range(0, len(adjMatrix)):
            if i == j and adjMatrix[i][j] != 0:
                print("Error in row: %d, column %d" % (i, j))
                print("The leading diagonal of the graph must be zeroes")
                raise customExceptions.MatrixHasNonZeroLeadingDiagonal
            if i != j and adjMatrix[i][j] == 0:
                print("Error in row %d, column %d" % (i, j))
                print("Non leading diagonal entries must not be zero (all destinations are considered to be connected)")
                raise customExceptions.MatrixHasZeroNotInLeadingDiagonal


def parseCustomGraph(fileName: string) -> Graph:
    """
    Parses the user submitted matrix and attempts to generate a graph
    :param fileName: The name of the user submitted file
    :return: The graph containing their matrix as its adjacency matrix
    """
    try:
        with open(fileName) as textFile:
            adjMatrix = [[int(x) for x in line.replace(',', '').replace('[', '').replace(']', '').split()]
                         for line in textFile]
        verifyValidGraph(np.array(adjMatrix, dtype=object))
        graph = Graph(len(adjMatrix))
        graph.addMatrix(adjMatrix)
        return graph
    except FileNotFoundError:
        raise
    except customExceptions.InputGraphInvalidException:
        raise


def validateUserArguments(args: List[str]) -> Tuple[int, Graph, bool]:
    """
    validates the user's command line arguments
    :param args: A list of arguments supplied by the user
    :return: A tuple containing information on the way the user wishes to run the program
    """
    requestsBruteForce = False
    if '-b' in args:
        requestsBruteForce = True
        args.remove('-b')

    if len(args) == 0:
        return 1, generateRandomGraph(), requestsBruteForce

    graph = parseUserInput(args[0])
    if len(args) == 2 and args[1].isnumeric():
        return int(args[1]), graph, requestsBruteForce
    else:
        return 1, graph, requestsBruteForce


def parseUserInput(arg):
    """
    Parses the users command line arguments if they are valid
    :param arg: The argument which is either a filename, or the number of cities requested in a random graph
    :return: The expected graph for their given arguments
    """
    if arg[-4:] == ".txt":
        try:
            return parseCustomGraph(arg)
        except FileNotFoundError:
            print("That file could not be found, please verify the file name and ensure the file can be reached")
            exit()
        except customExceptions.InputGraphInvalidException:
            print("An error has occurred generating the graph, please verify your matrix and try again!")
            exit()
        pass
    elif arg.isnumeric():
        return generateRandomGraph(int(arg))
    else:
        return generateRandomGraph()


def runAlgorithms(numberOfVehicles, graph, requestsBruteForce):
    """
    Runs each algorithm according to the user's command line arguments
    :param numberOfVehicles: The number of vehicles traversing the graph
    :param graph: An instance of our custom graph datastructure, containing the adjacency matrix
    :param requestsBruteForce: Whether the user requests to include the brute force algorithm
    """

    print("Simulating", numberOfVehicles, "vehicle(s) across the following adjacency matrix")
    print(graph.adjMatrix)

    # If there is only one vehicle, we run all the TSP algorithms, otherwise we are considering the VRP case.
    if numberOfVehicles == 1:
        if requestsBruteForce:
            determineExecutionTime(runBruteForceTSP, graph)
        determineExecutionTime(runNearestNeighbourTSP, graph)
        determineExecutionTime(runAntColonyTSP, graph)
        determineExecutionTime(runQUBOTSP, graph)
    else:
        if requestsBruteForce:
            determineExecutionTime(runBruteForceVRP, graph, numberOfVehicles)
        determineExecutionTime(runNearestNeighbourVRP, graph, numberOfVehicles)
        determineExecutionTime(runAntColonyVRP, graph, numberOfVehicles)
        determineExecutionTime(runQUBOVRP, graph, numberOfVehicles)


def main():
    numberOfVehicles, graph, requestsBruteForce = validateUserArguments(sys.argv[1:])
    runAlgorithms(numberOfVehicles, graph, requestsBruteForce)


if __name__ == '__main__':
    main()
