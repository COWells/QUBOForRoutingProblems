import numpy as np


def convertPathToQUBOTSPArray(path):
    """
    Converts a given TSP path to an Array which can be parsed by the QUBO
    :param path: A sequence of nodes the vehicle has travelled
    :return: The array which would generate that list
    """
    arraySize = len(path)
    quboArray = [[1 if j == path[i] else 0 for j in range(arraySize)] for i in range(arraySize)]
    return np.array(quboArray).transpose()


def convertPathToQUBOVRPArray(numberOfNodes, path, numberOfVehicles):
    """
    Converts a given list of VRP paths to an Array which can be parsed by the QUBO
    :param numberOfNodes: The number of nodes in the graph
    :param path: A sequence of nodes the vehicle has travelled
    :param numberOfVehicles: The number of vehicles traversing the graph
    :return: The array which would generate that list
    """
    quboArray = []
    for i in range(numberOfVehicles):
        vehiclePath = path[i]
        pathLength = len(vehiclePath)
        vehicleArray = [[1 if j == vehiclePath[k] else 0 for j in range(numberOfNodes)] for k in range(pathLength)]
        zeroes = np.zeros((numberOfNodes, numberOfNodes + 1 - pathLength))
        quboArray.append(np.hstack((np.array(vehicleArray).transpose(), zeroes)))
    return quboArray
