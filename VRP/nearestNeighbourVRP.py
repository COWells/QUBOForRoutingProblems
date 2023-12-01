from operator import attrgetter
from typing import List, Tuple
from GraphDataStructure import Graph


class Vehicle(object):
    """
    Class used to represent our vehicle
    currentNode: The node the vehicle is current stationed on
    pathTravelled: The path which the vehicle has travelled
    distanceTravelled: The total distance travelled by the vehicle
    """
    def __init__(self, startingNode):
        self.currentNode = startingNode
        self.pathTravelled = [startingNode]
        self.distanceTravelled = 0


def runNearestNeighbourVRP(graph: Graph, startingNode: int = 0,
                           numberOfVehicles: int = 2) -> Tuple[List[List[int]], int]:
    """
    Runs a custom variant of the nearest neighbour VRP algorithm on our graph
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :param startingNode: Our starting (and ending) node
    :param numberOfVehicles: The number of vehicles which will be traversing the graph
    :return: A tuple containing the route for each vehicle and the maximum distance travelled by a single vehicle
    """
    vehicles = [Vehicle(startingNode) for _ in range(numberOfVehicles)]
    numberOfNodes = graph.size
    visitedNodes = [startingNode]

    while len(visitedNodes) < numberOfNodes:

        # Select truck with minimum distance travelled.
        activeVehicle = min(vehicles, key=attrgetter('distanceTravelled'))

        # Creates an index, value dictionary for each edge of the current node.
        indexDict = dict()
        for i in range(len(graph.adjMatrix[activeVehicle.currentNode])):
            indexDict[i] = int(graph.adjMatrix[activeVehicle.currentNode][i])

        # Sorts the dictionary such that the values are in ascending order.
        # This means that the closest neighbour is the first element of the dictionary.
        indexDict = sorted(indexDict.items(), key=lambda x: x[1])

        for pair in indexDict:
            # Visit the node if we haven't visited it, otherwise check the next closest neighbour.
            if pair[0] not in visitedNodes:
                activeVehicle.currentNode = pair[0]
                visitedNodes.append(pair[0])
                activeVehicle.pathTravelled.append(pair[0])
                activeVehicle.distanceTravelled += pair[1]
                break

    # Manually add the distance returning from our last node back to our starting node.
    for vehicle in vehicles:
        vehicle.distanceTravelled += graph.adjMatrix[vehicle.currentNode][startingNode]
        vehicle.pathTravelled.append(startingNode)

    return [vehicle.pathTravelled for vehicle in vehicles], max(vehicle.distanceTravelled for vehicle in vehicles)