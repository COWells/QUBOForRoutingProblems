import math
import unittest
import numpy as np
import TSP
import VRP
import customExceptions
import main
import testUtils
from sympy.functions.combinatorial.numbers import stirling
from GraphDataStructure import Graph

adjMatrix = [[0.0, 3.0, 4.0, 2.0, 7.0],
             [3.0, 0.0, 4.0, 6.0, 3.0],
             [4.0, 4.0, 0.0, 5.0, 8.0],
             [2.0, 6.0, 5.0, 0.0, 6.0],
             [7.0, 3.0, 8.0, 6.0, 0.0]]


class TestBruteForce(unittest.TestCase):
    """
    Unit test class for the Brute Force Algorithm
    """

    # Make sure that the number of permutations generated is correct for the TSP case
    def test_permutation_amount(self):
        for i in range(3, 10):
            with self.subTest(i=i):
                nodesToVisit = [x for x in range(i) if x != 0]
                routePermutations = list(TSP.bruteForceTSP.generateRoutes(nodesToVisit))
                self.assertEqual(len(routePermutations), math.factorial(i - 1) / 2)

    # Make sure that given a graph, the correct path and path length is output in the TSP case
    def test_TSP_produce_minimum_path(self):
        graph = Graph(5)
        graph.adjMatrix = adjMatrix
        path, pathLength = TSP.runBruteForceTSP(graph)
        self.assertEqual(pathLength, 19)
        self.assertTrue(path == [0, 2, 1, 4, 3, 0] or path == [0, 3, 4, 1, 2, 0])

    # Make sure that the number of set partitions generated is correct for the VRP case
    def test_partition_amount(self):
        for i in range(3, 10):
            for n in range(2, 8):
                with self.subTest(i=i, n=n):
                    n = min(i - 1, n)
                    nodesToVisit = [int(x) for x in range(i) if x != 0]
                    routePartitions = list(VRP.bruteForceVRP.generateAllPartitions(nodesToVisit, n, n))
                    self.assertEqual(len(routePartitions), stirling(i - 1, n, kind=2))

    # Make sure that given a graph, the correct path and path length is output in the VRP case
    def test_VRP_produce_minimum_path(self):
        graph = Graph(5)
        graph.adjMatrix = adjMatrix
        path, pathLength = VRP.runBruteForceVRP(graph, numberOfVehicles=2)
        self.assertEqual(path, [[0, 2, 3, 0], [0, 1, 4, 0]])
        self.assertEqual(pathLength, 13)

        path, pathLength = VRP.runBruteForceVRP(graph, numberOfVehicles=3)
        self.assertEqual(path, [[0, 2, 0], [0, 3, 0], [0, 1, 4, 0]])
        self.assertEqual(pathLength, 13)


class TestQUBOTSPFormulation(unittest.TestCase):
    """
    Unit test class for the QUBO formulation for the TSP case
    """

    graph = Graph(5)
    graph.adjMatrix = adjMatrix

    correct_path = [0, 3, 4, 1, 2, 0]
    pathLength = 19
    correct_quboArray = testUtils.convertPathToQUBOTSPArray(correct_path[1:])

    bad_path = [0, 3, 4, 1, 2, 2]
    bad_quboArray = testUtils.convertPathToQUBOTSPArray(bad_path[1:])

    # Tests that given a graph, the correct cost function is output
    def test_cost_function_TSP(self):
        self.assertEqual(TSP.quboTSP.generateDistance(self.graph, self.correct_quboArray), self.pathLength)

    # Tests that given a valid array, the penalty functions are all 0
    def test_penalty_functions_for_good_path_TSP(self):
        nodeNum = self.graph.size
        constraint1 = sum(
            [sum((self.correct_quboArray[i][t] for t in range(nodeNum)), -1) ** 2 for i in range(nodeNum)])
        constraint2 = sum(
            [sum((self.correct_quboArray[i][t] for i in range(nodeNum)), -1) ** 2 for t in range(nodeNum)])
        constraint3 = (self.correct_quboArray[0][nodeNum - 1] - 1) ** 2
        self.assertTrue(constraint1 == 0 and constraint2 == 0 and constraint3 == 0)

    # Tests that given an invalid path, at least one of the penalty functions is not 0
    def test_penalty_functions_for_bad_path_TSP(self):
        nodeNum = self.graph.size
        constraint1 = sum([sum((self.bad_quboArray[i][t] for t in range(nodeNum)), -1) ** 2 for i in range(nodeNum)])
        constraint2 = sum([sum((self.bad_quboArray[i][t] for i in range(nodeNum)), -1) ** 2 for t in range(nodeNum)])
        constraint3 = (self.bad_quboArray[0][nodeNum - 1] - 1) ** 2
        self.assertTrue(constraint1 != 0 or constraint2 != 0 or constraint3 != 0)


class TestQUBOVRPFormulation(unittest.TestCase):
    """
    Unit test class for the QUBO formulation for the VRP case
    """
    graph = Graph(5)
    graph.adjMatrix = adjMatrix
    numberOfVehicles = 2

    correct_path = [[0, 3, 2, 0], [0, 4, 1, 0]]
    correct_quboArray = testUtils.convertPathToQUBOVRPArray(graph.size, correct_path, numberOfVehicles)

    bad_path = [[0, 3, 2, 2], [0, 0, 1, 1]]
    bad_quboArray = testUtils.convertPathToQUBOVRPArray(graph.size, bad_path, numberOfVehicles)

    # Tests that given a valid array, the penalty functions are all 0
    def test_static_penalty_functions_good_path_VRP(self):
        numberOfNodes = self.graph.size

        constraint1 = []
        for j in range(1, numberOfNodes):
            constraint1.append((sum((self.correct_quboArray[i][j][k] for k in range(numberOfNodes + 1)
                                     for i in range(self.numberOfVehicles)), -1) ** 2))
        constraint1 = sum(constraint1)

        constraint2 = []
        for i in range(1, self.numberOfVehicles):
            constraint2.append(sum((self.correct_quboArray[i][0][k] for k in range(numberOfNodes + 1)), -2) ** 2)
        constraint2 = sum(constraint2)

        constraint3 = []
        for i in range(self.numberOfVehicles):
            value = int(self.correct_quboArray[i][0][0])
            constraint3.append((value - 1) ** 2)
        constraint3 = sum(constraint3)

        constraint4 = []
        for i in range(self.numberOfVehicles):
            value = int(self.correct_quboArray[i][0][1])
            constraint4.append(value ** 2)
        constraint4 = sum(constraint4)

        self.assertTrue(constraint1 == 0 and constraint2 == 0 and constraint3 == 0 and constraint4 == 0)

    # Tests that given an invalid path, at least one of the penalty functions is not 0
    def test_static_penalty_functions_bad_path_VRP(self):
        numberOfNodes = self.graph.size

        constraint1 = []
        for j in range(1, numberOfNodes):
            constraint1.append((sum((self.bad_quboArray[i][j][k] for k in range(numberOfNodes + 1)
                                     for i in range(self.numberOfVehicles)), -1) ** 2))
        constraint1 = sum(constraint1)

        constraint2 = []
        for i in range(1, self.numberOfVehicles):
            constraint2.append(sum((self.bad_quboArray[i][0][k] for k in range(numberOfNodes + 1)), -2) ** 2)
        constraint2 = sum(constraint2)

        constraint3 = []
        for i in range(self.numberOfVehicles):
            value = int(self.bad_quboArray[i][0][0])
            constraint3.append((value - 1) ** 2)
        constraint3 = sum(constraint3)

        constraint4 = []
        for i in range(self.numberOfVehicles):
            value = int(self.bad_quboArray[i][0][1])
            constraint4.append(value ** 2)
        constraint4 = sum(constraint4)

        self.assertTrue(constraint1 != 0 or constraint2 != 0 or constraint3 != 0 or constraint4 != 0)


class TestMatrixParsing(unittest.TestCase):
    """
    Unit test class for the parsing of a users adjacency matrix
    """

    # Tests that given a valid matrix, it will create a valid graph.
    def test_matrix_parsed_correctly(self):
        graph = main.parseCustomGraph("customGraph.txt")
        self.assertEqual(graph.adjMatrix, adjMatrix)

    # Test that given a non-square matrix, it will throw the correct error
    def test_non_square_matrix_throws_exception(self):
        adjMatrix.append([1])
        matrix = np.array(adjMatrix, dtype=object)
        self.assertRaises(customExceptions.MatrixNotSquareException, main.verifyValidGraph, matrix)
        adjMatrix.pop()

    # Test that given a non-symmetric matrix, it will throw the correct error
    def test_non_symmetric_matrix_throws_exception(self):
        adjMatrix[3][1] = 99
        matrix = np.array(adjMatrix, dtype=object)
        self.assertRaises(customExceptions.MatrixNotSymmetricException, main.verifyValidGraph, matrix)
        adjMatrix[3][1] = 6

    # Test that given a matrix with non-zeroes in the leading diagonal, it will throw the correct error
    def test_non_zero_leading_diagonal_matrix_throws_exception(self):
        adjMatrix[0][0] = 1
        matrix = np.array(adjMatrix, dtype=object)
        self.assertRaises(customExceptions.MatrixHasNonZeroLeadingDiagonal, main.verifyValidGraph, matrix)
        adjMatrix[0][0] = 0

    # Test that given a matrix with zeroes not in the leading diagonal, it will throw the correct error
    def test_zero_not_in_leading_diagonal_matrix_throws_exception(self):
        adjMatrix[0][1] = adjMatrix[1][0] = 0
        matrix = np.array(adjMatrix, dtype=object)
        self.assertRaises(customExceptions.MatrixHasZeroNotInLeadingDiagonal, main.verifyValidGraph, matrix)
        adjMatrix[0][1] = adjMatrix[1][0] = 3


if __name__ == "__main__":
    unittest.main()
