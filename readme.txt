
This project was created for ECM3401: Individual Literature Review and Project by Charlie W.

This program contains various algorithms to solve the travelling salesman and vehicle routing problem.
It was designed to compare the accuracy and efficiency of QUBO formulations in comparison with three classical algorithms.
The included algorithms are: Brute Force, Nearest Neighbour, Ant Colony Optimisation, Simulated Annealing for solving QUBO formulation.

To run the program please ensure that all requirements (listed in requirements.txt) are succesfully installed.
Then call the main function as such:
python main.py [filename.txt]/[NumberOfCities] [NumberOfVehicles]

One can include command line arguments for custom inputs:
> The flag "-b" can be included anywhere and will cause the brute force algorithm to run, in addition to the other algorithms.
  (This algorithm is disabled by default due to its long runtime for large numbers of destinations).
  
> A filename can be included which reads in a custom adjacency matrix from a text file
  Note this adjacency matrix must match the expected criteria for TSP or VRP, that is, it must be Symmetric and Square,
  with 0s in the leading diagonal, and non-0 everywhere else).
  A custom graph has been included in the UnitTest folder, showing the expected format.
  
> Alternatively, one could include the number of cities for a randomly generated graph (instead of inputting a filename).
  If neither a filename nor the number of cities is included, a graph of size 10 will be randomly generated.
	
> An integer indicating the number of vehicles can be included immediately after the filename or number of cities.
  This converts the problem from TSP to VRP, which the program will handle automatically.
  

Examples:
python main.py 5 2
> This will randomly create a graph with 5 cities, and perform VRP on it with 2 vehicles.

python main.py "customFile.txt" 3
> This will import a custom graph from the supplied text file, then perform VRP on it with 3 vehicles.

python main.py 6 -b
> This will randomly create a graph with 6 cities, and perform TSP on it with 1 vehicle (as the numberOfVehicles was not supplied).
> This will also invoke the brute force algorithm to obtain an optimal solution, due to the -b flag included.