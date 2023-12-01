import math
from typing import Tuple, Dict, List
from collections import defaultdict
from neal import SimulatedAnnealingSampler
from pyqubo import Array, Constraint, DecodedSample, Model, Placeholder
from GraphDataStructure import Graph


def generatePossibleTourPermutations(numberOfNodes: int, numberOfVehicles: int) -> List[Tuple[int]]:
    """
    Given the numberOfNodes and numberOfVehicles, generate unique permutations
    which represent the amount of cities each vehicle visits.
    For instance, with 4 visiting nodes and 2 vehicles, this will return [(1, 3), (2, 2)].
    This represents two unique possibilities:
        The first vehicle visits 1 location then returns (and the other visits 3 locations)
        The first vehicle visits 2 locations then returns (and the other visits 2 locations).
    :param numberOfNodes: The number of cities in our graph
    :param numberOfVehicles: The number of vehicles that will be traversing the graph
    """
    results = []

    def getPermutations(total, n, start=1, end=None, prefix=()):
        if n == 0:
            if total == 0:
                results.append(prefix)
            return
        if end is None:
            end = total
        for i in range(start, end + 1):
            if i > total:
                break
            getPermutations(total - i, n - 1, i, end, prefix + (i,))

    getPermutations(numberOfNodes - 1, numberOfVehicles)
    return results


def generateCostFunction(graph: Graph, binaryArray: Array, numberOfVehicles: int) -> int:
    """
    Generates the cost function which is combined with the constraints to form the QUBO
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :param binaryArray: The array of binary values generated for our QUBO
    :param numberOfVehicles: The number of vehicles which will be traversing the graph
    :return: The calculation for the distance given our binaryArray.
    """
    distanceMatrix = graph.adjMatrix
    nodeNumber = graph.size
    vehicleDistances = []
    for i in range(numberOfVehicles):
        vehicleDistance = 0
        for j in range(nodeNumber):
            for k in range(nodeNumber):
                # For each entry, the distance is determined by the weight of each route taken.
                # Summing each element in this way means a row and column with one 1 has lower distance
                # This retroactively reduces cycles while still giving us the correct result.
                vehicleDistance += distanceMatrix[j][k] * sum(binaryArray[i][j][t] * binaryArray[i][k][(t + 1)]
                                                              for t in range(nodeNumber))
        vehicleDistances.append(vehicleDistance)
    return sum(vehicleDistances)


def generateStaticConstraints(numbersOfNodes: int, binaryArray: Array,
                              startingNode: int, numberOfVehicles: int) -> Tuple:
    """
    Generates the static constraints that are combined with the distance calculation to form part of the QUBO.
    Static constraints are constraints that do not change when the number of location each vehicle visits changes
    :param numbersOfNodes: The number of cities in our graph
    :param binaryArray: The array of binary values generated for our QUBO
    :param startingNode: The node we wish to start and end on
    :param numberOfVehicles: The number of vehicles which will be traversing the graph
    :return: A tuple of lists containing each constraint
    """

    # Constraint 1: Constrains the amount of times we visit each (non-starting) node.
    # Each row except the "starting row" can only be visited once by any vehicle.
    node_visited_once_list = []
    for j in range(1, numbersOfNodes):
        node_visited_once_list.append(Constraint(sum((binaryArray[i][j][k] for k in range(numbersOfNodes + 1)
                                                      for i in range(numberOfVehicles)), -1) ** 2,
                                                 label='node_visited_once'))
    node_visited_once_constraint = sum(node_visited_once_list)

    # Constraint 2: Constrains the amount of times we visit the starting node.
    # The starting node should be visited exactly twice by each vehicle (Once at the start, once when we return).
    starting_node_visited_twice_list = []
    for i in range(numberOfVehicles):
        starting_node_visited_twice_list.append(Constraint(sum((binaryArray[i][startingNode][k]
                                                                for k in range(numbersOfNodes + 1)), -2) ** 2,
                                                           label='starting_node_visited_twice'))
    starting_node_visited_twice_constraint = sum(starting_node_visited_twice_list)

    # Constraint 3: Constrains the node which the vehicles start at.
    # Every vehicle should start at the designated starting node.
    starts_at_starting_node_list = []
    for i in range(numberOfVehicles):
        starts_at_starting_node_list.append(Constraint((binaryArray[i][startingNode][0] - 1) ** 2,
                                                       label='starts_at_starting_node'))
    starts_at_starting_node_constraint = sum(starts_at_starting_node_list)

    # Constraint 4: Constrains that first action is not visiting starting node.
    # Every vehicle should immediately leave the starting node.
    leaves_starting_node_list = []
    for i in range(numberOfVehicles):
        leaves_starting_node_list.append(Constraint((binaryArray[i][startingNode][1]) ** 2,
                                                    label='leaves_starting_node'))
    leaves_starting_node_constraint = sum(leaves_starting_node_list)

    return node_visited_once_constraint, starting_node_visited_twice_constraint, \
        starts_at_starting_node_constraint, leaves_starting_node_constraint


def generateDynamicConstraints(numbersOfNodes: int, binaryArray: Array,
                               startingNode: int, numberOfVehicles: int, vehicleLocations: Tuple[int]) -> Tuple:
    """
    Generates the dynamic constraints that are combined with the distance calculation to form part of the QUBO.
    Dynamic constraints are constraints that change when the number of location each vehicle visits changes
    :param numbersOfNodes: The number of cities in our graph
    :param binaryArray: The array of binary values generated for our QUBO
    :param startingNode: The node we wish to start and end on
    :param numberOfVehicles: The number of vehicles which will be traversing the graph
    :param vehicleLocations: Tuple dictating how many destination each vehicle will visit
    :return: A tuple of lists containing each constraint
    """

    # Constraint 5: Constrains that each vehicle returns to the starting node after it has visited N locations
    ends_at_starting_node_list = []
    for i in range(numberOfVehicles):
        ends_at_starting_node_list.append(Constraint((binaryArray[i][startingNode][vehicleLocations[i] + 1] - 1) ** 2,
                                                     label='returns_to_starting_node'))
    ends_at_starting_node_constraint = sum(ends_at_starting_node_list)

    # Constraint 6: Constrains the order that a node is visited.
    # A single vehicle cannot visit two nodes simultaneously and must move to a new node during each action.
    node_order_meaningful_list = []
    for i in range(numberOfVehicles):
        for k in range(vehicleLocations[i] + 2):
            node_order_meaningful_list.append(
                Constraint(sum((binaryArray[i][j][k] for j in range(numbersOfNodes)), -1) ** 2,
                           label='node_order_meaningful'))
    node_order_meaningful_constraint = sum(node_order_meaningful_list)

    return ends_at_starting_node_constraint, node_order_meaningful_constraint


def determinePenaltyValue(graph: Graph) -> int:
    """
    Uses the MQC method to generate a penalty value by returning the greatest distance between any two nodes
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    :return: The maximum distance amongst all edge in the graph
    """
    return max(map(max, graph.adjMatrix))


def determineMaxVehicleDistance(sample: DecodedSample, distanceArray: List[List[int]]):
    """
    Determines the max vehicle distance from a given sample
    :param sample: The decoded sample alongside it's energy level
    :param distanceArray: Adjacency matrix storing the distances between each vertex
    :return: The maximal distance a single vehicle has travelled in the given sample.
    """

    def getDistance(x: int, y: int) -> int:
        return distanceArray[x][y]

    parsedRoutes = parseRoutes(sample)

    return max(sum(getDistance(route[i], route[i + 1]) for i in range(len(route) - 1)) for route in parsedRoutes)


def determineBestSample(samples: List[DecodedSample], distanceArray: List[List[int]]) -> Tuple:
    """
    Determines from our list of samples, which is the best according to our criteria
    (We are attempting to minimise the maximum distance travelled by any single vehicle).
    :param samples: List of decoded samples alongside their energy level.
    :param distanceArray: Adjacency matrix storing the distances between each vertex
    :return: The optimal route and the calculated energy level given our list of samples.
    """

    bestVehicleDistance = math.inf
    bestSample = None

    for sample in samples:
        vehicleRouteLength = determineMaxVehicleDistance(sample.sample, distanceArray)
        if vehicleRouteLength < bestVehicleDistance:
            bestVehicleDistance = vehicleRouteLength
            bestSample = sample

    return bestSample, bestVehicleDistance


def simulatedAnnealing(model: Model, feedDict: Dict[str, float], numberOfVehicles: int) -> List[DecodedSample]:
    """
    Uses the neal module to perform Simulated Annealing on our model
    The values of the placeholders are specified by our feedDict dictionary
    :param model: The model we compiled by using our Hamiltonian
    :param feedDict: The dictionary specifying the values of our placeholders
    :param numberOfVehicles: The number of vehicles which will be traversing the graph
    :return: The binary array which resulted in the lowest energy level
    """

    # Instantiate our Simulated Annealer and binary quadratic model
    sa = SimulatedAnnealingSampler()
    bqm = model.to_bqm(feed_dict=feedDict)

    # Perform simulated annealing on our binary quadratic model
    sampleSet = sa.sample(bqm, num_reads=500)

    decodedSamples = model.decode_sampleset(sampleSet, feedDict)

    # Return all samples that could be the correct solution.
    # (It is guaranteed that any energy level which is greater than N times higher than our minimum energy,
    # must contain at least one vehicle which travels further than our minimum sum, where N is the number of vehicles).
    return [sample for sample in decodedSamples if sample.energy < numberOfVehicles * decodedSamples[0].energy
            and len(sample.constraints(only_broken=True)) == 0]


def parseRoutes(bestRoute: DecodedSample) -> List[List[int]]:
    """
    Parses our result into a readable form which we can display to the user
    :param bestRoute: The route we obtained with the lowest energy level
    :return: A list containing each vehicle's route, formatted such that it can be printed properly.
    """
    routeList = [route[0] for route in bestRoute.items() if route[1] == 1]
    # Create a dictionary, each key represents a vehicle's number and contains the full path of that vehicle.
    vehicleRoutes = defaultdict(list)
    for x in routeList:
        x = x.translate(str.maketrans("", "", "x[")).split(']')[:-1]
        vehicleRoutes[x[0]].append(tuple(x[1:]))
    # Parses the routes so that it is indexed according to the order that the vehicle visited the node
    parsedRoutes = []
    for route in vehicleRoutes.values():
        route.sort(key=lambda z: int(z[1]))
        parsedRoutes.append([int(z[0]) for z in route])

    return parsedRoutes


def runQUBOVRP(graph: Graph, startingNode: int = 0, numberOfVehicles: int = 2):
    """
    Runs QUBO TSP on our graph through the use of Simulated Annealing
    :param numberOfVehicles: The number of vehicles which will be traversing the graph
    :param startingNode: The node at which all vehicles will start and end on
    :param graph: Uses our custom Graph class which contains the adjacency matrix
    """
    Penalty = Placeholder("P")

    numberOfNodes = graph.size
    if numberOfVehicles >= numberOfNodes:
        print("You have more vehicles than non-depot locations...")
        print("Considering the case where you have", numberOfNodes - 1, "vehicles instead!")
        numberOfVehicles = numberOfNodes - 1

    # Generate the constraints and distance calculations for our Hamiltonian.
    binaryArray = Array.create('x', shape=(numberOfVehicles, numberOfNodes, numberOfNodes + 1), vartype='BINARY')

    # Generate the static constraints (those that do not rely on knowing the number of locations each vehicle visits)
    staticConstraints = generateStaticConstraints(numberOfNodes, binaryArray, startingNode, numberOfVehicles)

    # Generate the cost function (which calculates the distance of each vehicle's path)
    costFunction = generateCostFunction(graph, binaryArray, numberOfVehicles)

    penaltyValue = determinePenaltyValue(graph)**2

    globalMinimumEnergy = math.inf
    globalMinimumRoute = None
    for permutation in generatePossibleTourPermutations(numberOfNodes, numberOfVehicles):

        # Generate the dynamic constraints (those that rely on knowing the number of locations each vehicle visits)
        dynamicConstraints = generateDynamicConstraints(numberOfNodes, binaryArray, startingNode,
                                                        numberOfVehicles, permutation)

        H = costFunction + Penalty * (sum(constraint for constraint in staticConstraints)) + \
            Penalty * (sum(constraint for constraint in dynamicConstraints))

        model = H.compile()
        samples = simulatedAnnealing(model, {'P': penaltyValue}, numberOfVehicles)
        bestRoute, bestEnergy = determineBestSample(samples, graph.adjMatrix)
        # Check whether this permutation is better than previous permutations
        if bestEnergy < globalMinimumEnergy:
            globalMinimumEnergy = bestEnergy
            globalMinimumRoute = bestRoute

    return parseRoutes(globalMinimumRoute.sample), globalMinimumEnergy
