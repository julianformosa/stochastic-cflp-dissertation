# stochastic-cflp-dissertation

This repository holds the scripts used to produce the results for my undergraduate thesis, "Solving the Stochastic Competitive Facility Location Problem using Metaheuristic Algorithms."

```generate_instances.py``` is used to generate problem instances based on the EV charging station industry in Sliema, Malta.
`exact_method.py` is used to solve the instances using an exact method.
`master_alg.py` is used to solve the instances using the Master Algorithm (a memetic algorithm composed of Genetic Algorithm and Tabu Search).
`market_potential_alg.py` is used to solve the instances using the Market Potential Algorithm (a heuristic method).
`tuning.py` is used to perform parameter tuning via random search on the Master Algorithm's parameters.
`testing.py` is used to compare the performance of exact methods, the Master Algorithm, and the Market Potential Algorithm.
