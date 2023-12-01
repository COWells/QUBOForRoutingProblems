from typing import Tuple, Dict, List
from neal import SimulatedAnnealingSampler
from pyqubo import Array, Constraint, DecodedSample, Model, Placeholder
from GraphDataStructure import Graph


def generateDistance(graph: Graph, binaryArray: Array) -> int:
    """
    Generates the distance calculation which is combined with the constraints to form the QUBO
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :param binaryArray: The array of binary values generated for our QUBO
    :return: The calculation for the distance given our binaryArray.
    """
    distanceMatrix = graph.adjMatrix
    nodeNumber = graph.size
    distance = 0
    for i in range(nodeNumber):
        for j in range(nodeNumber):
            # For each entry, the distance is determined by the weight of each route taken.
            # Summing each element in this way means a row and column with one 1 has lower distance
            # This retroactively reduces cycles while still giving us the correct result.
            distance += distanceMatrix[i][j] * sum(binaryArray[i][t] * binaryArray[j][(t + 1) % nodeNumber]
                                                   for t in range(nodeNumber))
    return distance


def generateConstraints(numbersOfNodes: int, binaryArray: Array,
                        startingNode: int = 0) -> Tuple:
    """
    Generates the constraints which are combined with the distance calculation to form the QUBO
    :param numbersOfNodes: The number of cities in our graph
    :param binaryArray: The array of binary values generated for our QUBO
    :param startingNode: The node we wish to start and end on
    :return: A tuple of lists containing each constraint
    """

    # Constraint 1: Constrains the amount of times we visit each node.
    # Each node should be visited exactly once.
    node_visited_once = []
    for i in range(numbersOfNodes):
        node_visited_once.append(Constraint(sum((binaryArray[i][t] for t in range(numbersOfNodes)), -1) ** 2,
                                            label='node_visited_once'))

    # Constraint 2: Constrains the order that a node is visited (1st, 2nd, 3rd etc...)
    # Each node should have an order, and this order should be unique for the node.
    order_appears = []
    for t in range(numbersOfNodes):
        order_appears.append(Constraint(sum((binaryArray[i][t] for i in range(numbersOfNodes)), -1) ** 2,
                                        label='order_appears'))

    # Constraint 3: We should finish at the node we designated as the starting node.
    arrive_last = Constraint((binaryArray[startingNode][numbersOfNodes - 1] - 1) ** 2, label='finish_at_starting_node')

    return order_appears, node_visited_once, arrive_last


def simulatedAnnealing(model: Model, feedDict: Dict[str, float]) -> DecodedSample:
    """
    Uses the neal module to perform Simulated Annealing on our model
    The values of the placeholders are specified by our feedDict dictionary
    :param model: The model we compiled by using our Hamiltonian
    :param feedDict: The dictionary specifying the values of our placeholders.
    :return: The binary array which resulted in the lowest energy level
    """

    # Instantiate our Simulated Annealer and binary quadratic model
    sa = SimulatedAnnealingSampler()
    bqm = model.to_bqm(feed_dict=feedDict)

    # Perform simulated annealing on our binary quadratic model
    sampleSet = sa.sample(bqm, beta_range=[0.1, 4],
                          num_reads=1500, beta_schedule_type='geometric')
    decodedSamples = model.decode_sampleset(sampleSet, feedDict)

    # Return the sample which has the minimum energy
    bestSample = min(decodedSamples, key=lambda z: z.energy)
    return bestSample


def parseBestSample(bestSample: DecodedSample) -> List[int]:
    """
    Parses our result into a readable form which we can display to the user
    :param bestSample: The sample we obtained with the lowest energy level
    :return: The route which should be taken as a list
    """
    resultDict = bestSample.sample
    routeList = [k for k, v in resultDict.items() if v == 1]

    # routeList contains a value which has the shape x[y][z].
    # We are sorting each element based on the value of z.
    routeList.sort(key=lambda z: int(z.split('[')[2][:-1]))

    # Now the list is sorted, we want to create a list of nodes visited, so we extract "y" from each element.
    # And prepend the final node to display the full path as opposed to a cycle.
    route = [int(y.split('[')[1][:-1]) for y in routeList]
    route.insert(0, route[-1])
    return route


def determinePenaltyValue(graph: Graph) -> int:
    """
    Uses the MQC method to generate a penalty value by returning the greatest distance between any two nodes
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :return: The maximum distance amongst all edge in the graph
    """
    return max(map(max, graph.adjMatrix))


def runQuboTSP(graph: Graph):
    """
    Runs QUBO TSP on our graph through the use of Simulated Annealing
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    """
    numberOfNodes = graph.size
    Penalty = Placeholder("P")

    # Generate the constraints and distance calculations for our Hamiltonian.
    binaryArray = Array.create('x', shape=(numberOfNodes, numberOfNodes), vartype='BINARY')
    distance = generateDistance(graph, binaryArray)
    order_once, visit_once, arrive_last = generateConstraints(numberOfNodes, binaryArray)

    # Concatenate the distance calculation and each constraint together and compile them to form our model
    H = distance + Penalty * (sum(order_once) + sum(visit_once) + arrive_last)
    model = H.compile()

    penaltyValue = determinePenaltyValue(graph)
    bestSample = simulatedAnnealing(model, {'P': penaltyValue})

    # Pick the result which violates none of the constraints (and pick the smallest of those).
    route = parseBestSample(bestSample)
    return route, bestSample.energy
