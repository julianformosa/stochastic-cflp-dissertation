""" Solves the Master Program using the Master Algorithm """


from generate_instances import *
import random
from collections import Counter


# SET PARAMETERS #######################################################################
# The parameters used were obtained using the parameter tuning process

PI_1 = 10_000
PI_2 = 10_000
POP_SIZE = 100
MAX_ITERATIONS = 3
MIN_IDENTICAL_INDIVIDUALS = 75
SCENARIO_COPY_RATE = 1

ITERATIONS_GA = 3
NUM_PARENT_PAIRS = 25
TOURNAMENT_SIZE = 30
MUTATION_PROBABILITY = 0.001
NUM_CHILDREN = 3
ELITISM_RATE = POP_SIZE - NUM_CHILDREN * NUM_PARENT_PAIRS
SELECTION_METHOD = "roulette"  # Can be either "truncation", "roulette", or "tournament"

TS_RATE = 1
ITERATIONS_TS = 5
NUM_NEIGHBOURS = 5
TENURE = 2


def stopping_condition(pop: list, iteration: int) -> bool:
    """ Checks whether the stopping condition has been reached """
    
    if iteration >= MAX_ITERATIONS:
        return True
    
    pop_as_strings = [dict_to_string_leader(k) for k in pop]  # Store individuals as strings
    
    # Count the number of times the most common individual appears
    counts = Counter(pop_as_strings)
    identical_individuals_count = counts.most_common()[0][1]
    
    if identical_individuals_count >= MIN_IDENTICAL_INDIVIDUALS:
        return True
    
    return False


# Initialise dictionaries to store the fitnesses of actions
fitness_follower_dict = dict()
fitness_leader_dict = dict()


def fitness_follower(k_follower: dict, k_leader: dict, s: int) -> float:
    """
    The fitness of Follower's action k_follower when Leader plays k_leader under scenario s
    """
    
    instance_string = dict_to_string_follower(k_follower, k_leader, s)
    
    try:  # Assume the fitness is already known
        return fitness_follower_dict[instance_string]
    
    except KeyError:  # Fitness of this action has not yet been calculated
        term_1 = profit_k("follower", k_leader, k_follower)
        term_2 = PI_1 * chi_0(
            max(0, cost_total_k("follower", k_leader, k_follower) - BUDGETS_FOLLOWER[s])
        )
        fitness_follower_dict[instance_string] = term_1 - term_2  # Store answer for future
        return term_1 - term_2


def fitness_leader(k_leader: dict) -> float:
    """ Evaluates the quality of Leader's action """
    
    instance_string = dict_to_string_leader(k_leader)
    
    try:  # Assume the fitness is already known
        return fitness_leader_dict[instance_string]
    
    except KeyError:  # Fitness of this action has not yet been calculated
        
        # Find Follower's corresponding strategy
        strategy = find_follower_strategy_MA(k_leader)
        
        # Calculate the fitness
        term_1 = master_objective_fun_k(
            k_leader, {s: strategy[s] for s in scenarios}
        )
        probability_obeying_budget = sum(
            SCENARIO_PROBABILITIES[s] * chi_k(k_leader, strategy[s])
            for s in scenarios
        )
        term_2 = PI_2 * chi_0(
            max([0, RHO - probability_obeying_budget])
        )
        fitness_leader_dict[instance_string] = term_1 - term_2  # Store answer for future
        return term_1 - term_2


def find_fitness(evaluated_action: dict, player: str, k_leader: dict, s: int) -> float:
    """
    Finds the fitness of evaluated_action.
    
    player is the one who played evaluated_action.
    If player is Follower, k_leader is Leader's action and s is the scenario.
    """
    
    # Calculate fitness depending on who player is
    if player == "leader":
        return fitness_leader(evaluated_action)
    else:  # player == "follower"
        return fitness_follower(evaluated_action, k_leader, s)


# Genetic Algorithm ####################################################################


def truncation(pop: list, player: str, k_leader: dict, s: int) -> list:
    """
    Performs truncation selection on the population pop of individuals.
    
    player is Leader if pop contains Leader's actions, and Follower if pop contains
    Follower's actions.
    If player is Follower, k_leader is Leader's action and s is the scenario.
    """
    
    # Sort pop in order of fitness
    sorted_pop = sorted(pop, key=lambda l: find_fitness(l, player, k_leader, s))
    
    # Select the best actions to form the parent pool
    parent_pool = sorted_pop[: 2 * NUM_PARENT_PAIRS]
    
    # Match up the parents in pairs
    parents = []
    while len(parents) < NUM_PARENT_PAIRS:
        [parent_1, parent_2] = random.sample(parent_pool, 2)
        parent_pool.remove(parent_1)
        parent_pool.remove(parent_2)
        parents.append([parent_1, parent_2])
    return parents


def roulette(pop: list, player: str, k_leader: dict, s: int) -> list:
    """
    Performs roulette wheel selection on the population pop of individuals.

    player is Leader if pop contains Leader's actions, and Follower if pop contains
    Follower's actions.
    If player is Follower, k_leader is Leader's action and s is the scenario.
    """

    parents = []
    parent_pool = pop.copy()
    
    # Use the fitness of each action as weights
    weights = [find_fitness(k, player, k_leader, s) for k in parent_pool]

    # Remove negative weights by adding a constant to every weight
    if any(w < 0 for w in weights):
        most_negative_weight = min(weights)
        weights = [w - most_negative_weight + 1 for w in weights]

    # Select the pairs of parents
    while len(parents) < NUM_PARENT_PAIRS:
        parent_1 = random.choices(parent_pool, weights)[0]
        parent_1_index = parent_pool.index(parent_1)
        weights.pop(parent_1_index)
        parent_pool.remove(parent_1)
        parent_2 = random.choices(parent_pool, weights)[0]
        parent_2_index = parent_pool.index(parent_2)
        weights.pop(parent_2_index)
        parent_pool.remove(parent_2)
        parents.append([parent_1, parent_2])
    return parents


def tournament(pop: list, player: str, k_leader: dict, s: int) -> list:
    """
    Performs tournament selection on the population pop of individuals.
    
    player is Leader if pop contains Leader's actions, and Follower if pop contains
    Follower's actions.
    If player is Follower, k_leader is Leader's action and s is the scenario.
    """
    
    parents = []
    parent_pool = pop.copy()
    while len(parents) < NUM_PARENT_PAIRS:
        
        # Generate tournament
        tournament_list = random.sample(parent_pool, TOURNAMENT_SIZE)
        
        # Pick the top two individuals to be a pair of parents
        sorted_tournament = sorted(
            tournament_list, key=lambda l: find_fitness(l, player, k_leader, s)
        )
        parent_pair = sorted_tournament[:2]
        parents.append(parent_pair)
        
    return parents


def recombination(k_1: dict, k_2: dict) -> dict:
    """ Combines the parents k_1 and k_2 into a new offspring """
    
    offspring = {v: 0 for v in vertices_unoccupied}
    
    vertices_1 = {v for v in vertices_unoccupied if k_1[v] != 0}
    vertices_2 = {v for v in vertices_unoccupied if k_2[v] != 0}
    
    # Fill in offspring's genes where both parents have built a facility
    for v in vertices_1 & vertices_2:
        offspring[v] = random.choice([k_1[v], k_2[v]])
    
    # Fill in offspring's genes where exactly one parent has built a facility
    vertices_available = random.sample(
        
        # List of vertices where exactly one parent has built a facility
        list(vertices_1 ^ vertices_2),
        
        # Only sample as many vertices as are needed to make sure the offspring has the
        # same amount of built facilities as the first parent
        sum(v != 0 for v in vertices_1) - len(vertices_1 & vertices_2)
    )
    for v in vertices_available:
        # Exactly one parent's allele will be 0, and the other parent's allele will be positive
        # The offspring's gene takes the value of the positive allele
        offspring[v] = max(k_1[v], k_2[v])
    
    return offspring


def mutation(k: dict, player: str, k_leader) -> dict:
    """
    Performs mutation on the action k with probability according to the mutation rate.
    
    player is the one who plays k.
    If player is Follower, k_leader is Leader's action.
    """
    
    m_1 = random.random()
    
    if m_1 < MUTATION_PROBABILITY:
        # Action selected for mutation
        
        feasible = False
        
        while not feasible:
            
            # Randomly decide what kind of mutation to use
            m_2 = random.choice([1, 2, 3, 4])
            
            try:
                if m_2 == 1:
                    # Make a 0-valued allele positive
                    v = random.choice([v for v in vertices_unoccupied if k[v] == 0])
                    k[v] = random.choice(range(1, MAX_QUALITY_INDICES[v] + 1))
                
                elif m_2 == 2:
                    # Make a positive-valued allele 0
                    v = random.choice([v for v in vertices_unoccupied if k[v] != 0])
                    k[v] = 0
                
                elif m_2 == 3:
                    # Change the quality level on one vertex
                    v = random.choice([v for v in vertices_unoccupied if k[v] != 0])
                    k[v] = random.choice(range(1, MAX_QUALITY_INDICES[v] + 1))
                
                else:  # m_2 == 4
                    # Swap two positive alleles
                    v = random.choice(vertices_unoccupied)
                    u = random.choice([u for u in vertices_unoccupied if u != v])
                    k[v], k[u] = k[u], k[v]
                
                # Check if the offspring remains feasible
                feasible = obeys_building_constraints_k(k, player, k_leader)
            
            except IndexError:  # Mutation failed for whatever reason
                # A new mutation will be done
                feasible = False
    
    return k


def genetic_algorithm(pop_old: list, player: str, k_leader: dict, s: int) -> list:
    """
    Performs one iteration of Genetic Algorithm to pop_old.
     
     player is Leader if pop_old contains Leader's actions, and player is Follower if
     pop_old contains Follower's actions.
     If player is Follower, k_leader is Leader's action and s is the scenario.
     """
    
    # Select the list of parents
    if SELECTION_METHOD == "truncation":
        parents = truncation(pop_old, player, k_leader, s)
    elif SELECTION_METHOD == "roulette":
        parents = roulette(pop_old, player, k_leader, s)
    else:  # SELECTION_METHOD == "tournament"
        parents = tournament(pop_old, player, k_leader, s)
    
    # Generate offspring from the parent pairs
    pop_new = []
    children_created = 0
    while children_created < NUM_CHILDREN:
        for [parent_1, parent_2] in parents:
            offspring = recombination(parent_1, parent_2)
            offspring = mutation(offspring, player, k_leader)  # Perform mutation process
            pop_new.append(offspring)
        children_created += 1
    
    # Perform elitism process
    sorted_pop_old = sorted(
        pop_old, key=lambda l: find_fitness(l, player, k_leader, s), reverse=True
    )
    for k in sorted_pop_old[:ELITISM_RATE]:
        pop_new.append(k)
    
    return pop_new


# Tabu Search ##########################################################################


def generate_neighbourhood(k_current: dict, player: str, k_leader: dict) -> list:
    """
    Generates the neighbours of the action k_current.
     
     player is the one playing k_current.
     If player is Follower, k_leader is Leader's action.
     """
    
    neighbours = []
    
    while len(neighbours) < NUM_NEIGHBOURS:
        
        k_neighbour = k_current.copy()
        
        # Choose the vertices which can have a different allele in the neighbour
        if player == "leader":
            possible_vertices = vertices_unoccupied
        else:  # player == "follower"
            # Follower can only change a facility at a vertex not chosen by Leader
            possible_vertices = [u for u in vertices_unoccupied if k_leader[u] == 0]
        
        if possible_vertices:
            # Possible for a neighbour to exist
            
            # Randomly choose a vertex where the allele will be different
            v_diff = random.choice(possible_vertices)
            
            # The new allele must not be restricted, and must be different from original
            possible_quality_indices = [
                q
                for q in range(MAX_QUALITY_INDICES[v_diff] + 1)
                if k_current[v_diff] != q
            ]
            
            q_diff = random.choice(possible_quality_indices)
            k_neighbour[v_diff] = q_diff
            
            # Only accept neighbour if it obeys the building constraints
            if obeys_building_constraints_k(k_neighbour, player, k_leader):
                neighbours.append(k_neighbour)
        
        else:  # All vertices taken by Leader
            return [k_neighbour]  # Only neighbour is the same action (build nothing)
    
    return neighbours


def best_from_list(lst: list, player: str, k_leader: dict, s: int) -> dict:
    """
    Selects the best action from a list of actions using the appropriate fitness function.
    
    lst is the list of action.
    player is the one the actions in lst belong to.
    If player is Follower, k_leader is Leader's action and s is the scenario.
    """
    
    # Store results from first action in lst
    best_action = lst[0].copy()
    best_fitness = find_fitness(lst[0], player, k_leader, s)
    
    # Check the remaining actions to see if any are better
    for k in lst:
        k_fitness = find_fitness(k, player, k_leader, s)
        if best_fitness < k_fitness:
            best_fitness = k_fitness
            best_action = k.copy()
    
    return best_action


def select_neighbour(
    neighbourhood: list,
    k_current: dict,
    f_best: float,
    tabu_list: list,
    player: str,
    k_leader: dict,
    s: int
) -> dict:
    """
    Selects the fittest neighbour of an action.
     
     neighbourhood is the list of the action's neighbours.
     k_current is the action for which we want to select a neighbour.
     f_best is the best fitness of any action found through the TS process.
     tabu_list is a list of location-quality pairs that are tabu.
     player is the one who plays k_current.
     If player is Follower, k_leader is Leader's action and s is the scenario.
     """
    
    # Sort neighbours into tabu vs valid
    tabu_neighbours = []
    valid_neighbours = []
    for k_neighbour in neighbourhood:
        
        v_diff = -100  # This should become set to a different number by the next 4 lines
        for v in vertices_unoccupied:
            if k_current[v] != k_neighbour[v]:
                v_diff = v
                break
        if [v_diff, k_neighbour[v_diff]] in tabu_list:
            tabu_neighbours.append(k_neighbour)
        else:
            valid_neighbours.append(k_neighbour)
    
    # Choose the best neighbour
    try:  # Assume at least one neighbour is not tabu
        best_valid_action = best_from_list(valid_neighbours, player, k_leader, s)
    except IndexError:  # All neighbours are tabu
        best_tabu_action = best_from_list(tabu_neighbours, player, k_leader, s)
        return best_tabu_action
    
    try:  # Assume at least one neighbour is tabu
        best_tabu_action = best_from_list(tabu_neighbours, player, k_leader, s)
    except IndexError:  # No neighbours are tabu
        return best_valid_action
    
    # Some neighbours are tabu and some neighbours are valid
    if f_best < find_fitness(best_tabu_action, player, k_leader, s):
        # best_tabu_action is better than the best fitness found so far
        return best_tabu_action
    else:  # Aspiration criterion not satisfied
        return best_valid_action


def tabu_search(k_old: dict, player: str, k_leader: dict, s: int) -> dict:
    """
    Applies several iterations of TS to an action.
    
    k_old is the action undergoing improvement by TS.
    player is the one who plays k_old.
    If player is Follower, k_leader is Leader's action and s is the scenario.
    """
    
    k_current = k_old
    k_best = k_old
    f_best = find_fitness(k_old, player, k_leader, s)
    
    tabu_list = []
    iteration = 0
    while iteration < ITERATIONS_TS:
        
        # Select a neighbour to replace the current action
        neighbours = generate_neighbourhood(k_current, player, k_leader)
        best_neighbour = select_neighbour(
            neighbours, k_current, f_best, tabu_list, player, k_leader, s
        )
        
        # Put the changed location-quality pair in the tabu list
        v_diff = -100  # This should become set to a different number by the next 4 lines
        for v in vertices_unoccupied:
            if k_current[v] != best_neighbour[v]:
                v_diff = v
                break
        tabu_list.append([v_diff, k_current[v_diff]])
        
        # Check if the new action is better than the best action found so far
        if f_best < find_fitness(best_neighbour, player, k_leader, s):
            f_best = find_fitness(best_neighbour, player, k_leader, s)
            k_best = best_neighbour.copy()
        
        if iteration > TENURE:
            # Remove oldest location-quality pair from the tabu list
            tabu_list.pop(0)
        
        k_current = best_neighbour
        iteration += 1
    
    return k_best


# Follower's Algorithm #################################################################


def generate_follower(k_leader: dict) -> dict:
    """ Generates a random Follower action when Leader plays k_leader """
    
    k_follower = {v: 0 for v in vertices_unoccupied}
    
    # Create a list of vertices on which Leader has not built any facilities
    available_vertices = [v for v in vertices_unoccupied if k_leader[v] == 0]
    
    # Calculate the maximum number of facilities that follower can build
    tau_f = min(
        N - num_existing_follower,
        num_vertices_unoccupied - num_facilities_k(k_leader)
    )
    
    # Select which vertices to build a facility on
    selected_vertices = random.sample(available_vertices, tau_f)
    
    # Select the quality levels with which to build the facilities
    for v in selected_vertices:
        q = random.choice(list(range(1, MAX_QUALITY_INDICES[v] + 1)))
        k_follower[v] = q
        
    return k_follower


# This dictionary will store Follower's optimal strategies for different Leader actions
follower_strategy_table = dict()


def find_follower_strategy_MA(k_leader: dict) -> dict:
    """ Finds Follower's corresponding strategy when Leader plays k_leader """
    
    try:  # Assume the corresponding strategy has already been found
        return follower_strategy_table[dict_to_string_leader(k_leader)]
    
    except KeyError:  # We have not yet found Follower's corresponding strategy
        
        # Check if Leader has built a facility on all unoccupied vertices
        if all(k_leader[v] != 0 for v in vertices_unoccupied):
            # Follower has nowhere available to build any facilities under any scenario
            return {s: {v: 0 for v in vertices_unoccupied} for s in scenarios}
        
        follower_actions = []
        strategy = dict()
        
        for s in scenarios:
            
            # Generate initial population
            while len(follower_actions) < POP_SIZE:
                k_follower = generate_follower(k_leader)
                follower_actions.append(k_follower)
            
            iteration = 0
            
            # Perform Follower's Algorithm ---------------------------------------------
            
            while not stopping_condition(follower_actions, iteration):
                
                # Genetic Algorithm
                for _ in range(ITERATIONS_GA):
                    follower_actions = genetic_algorithm(
                        follower_actions, "follower", k_leader, s
                    )
                
                # Tabu Search
                follower_actions = list(
                    sorted(
                        follower_actions,
                        key=lambda l: fitness_follower(l, k_leader, s),
                        reverse=True
                    )
                )
                for _ in range(TS_RATE):
                    action = follower_actions[0].copy()
                    follower_actions.pop(0)  # Remove old action
                    improved_action = tabu_search(action, "follower", k_leader, s)
                    follower_actions.append(improved_action)  # Add in improved action
                
                iteration += 1
            
            # Choosing the best action -------------------------------------------------
            
            # Store results from the empty action
            best_feasible_action = {v: 0 for v in vertices_unoccupied}
            best_feasible_fitness = 0
            is_feasible_check = {
                dict_to_string_follower(best_feasible_action, k_leader, s): True
            }
            
            # Check remaining actions in final population to see if any are better
            for k in follower_actions:
                current_fitness = fitness_follower(k, k_leader, s)
                if current_fitness > best_feasible_fitness:
                    # Fittest action so far, it is worth checking for feasibility
                    
                    try:  # Assume we already know whether the action is feasible
                        if is_feasible_check[dict_to_string_follower(k, k_leader, s)]:
                            # New best action found
                            best_feasible_fitness = current_fitness
                            best_feasible_action = k
                    except KeyError:  # The action has not yet been checked for feasibility
                        if cost_total_k("follower", k_leader, k) <= BUDGETS_FOLLOWER[s]:
                            # New best action found
                            best_feasible_fitness = current_fitness
                            best_feasible_action = k
                            is_feasible_check[dict_to_string_follower(k, k_leader, s)] = True
                        else:
                            # Action is infeasible
                            is_feasible_check[dict_to_string_follower(k, k_leader, s)] = False
            
            strategy[s] = best_feasible_action
            
            # Copy over the best actions to warm-start the next scenario ---------------
            
            sorted_follower_actions = list(
                sorted(
                    follower_actions,
                    key=lambda l: fitness_follower(l, k_leader, s),
                    reverse=True
                )
            )
            follower_actions = sorted_follower_actions[:SCENARIO_COPY_RATE]
            # --------------------------------------------------------------------------
            
        follower_strategy_table[dict_to_string_leader(k_leader)] = strategy
        
        return strategy


# Master Algorithm #####################################################################


def generate_leader() -> dict:
    """ Generates a random Leader action """
    k = {v: 0 for v in vertices_unoccupied}
    
    # Calculate the maximum number of facilities Leader can build
    tau_l = min(N - num_existing_leader, num_vertices_unoccupied)
    
    # Randomly select vertices for Leader to build facilities on
    selected_vertices = random.sample(vertices_unoccupied, tau_l)
    
    # Randomly select the quality levels with which to build facilities
    for v in selected_vertices:
        q = random.choice(range(1, MAX_QUALITY_INDICES[v] + 1))
        k[v] = q
        
    return k


def find_leader_MA() -> dict:
    """ Finds Leader's optimal action """
    
    leader_actions = []
    
    # Generate initial population
    while len(leader_actions) < POP_SIZE:
        k_leader = generate_leader()
        leader_actions.append(k_leader)
    
    iteration = 0
    
    # Perform Master Algorithm ---------------------------------------------------------
    
    while not stopping_condition(leader_actions, iteration):
        print(f"Leader iteration {iteration}")  # Monitor progress for debugging purposes
        
        # Genetic Algorithm
        for i in range(ITERATIONS_GA):
            print(f"GA iteration {i}")  # Monitor progress for debugging purposes
            leader_actions = genetic_algorithm(leader_actions, "leader", None, None)
        
        # Tabu Search
        leader_actions = list(
            sorted(
                leader_actions,
                key=lambda l: fitness_leader(l),
                reverse=True
            )
        )
        for j in range(TS_RATE):
            print(f"TS improvement {j}")  # Monitor progress for debugging purposes
            action = leader_actions[0].copy()
            leader_actions.pop(0)
            improved_action = tabu_search(action, "leader", None, None)
            leader_actions.append(improved_action)
            
        iteration += 1
    
    # Choosing the best action ---------------------------------------------------------
    
    # Store results from empty action
    best_feasible_action = {v: 0 for v in vertices_unoccupied}
    best_feasible_fitness = 0
    is_feasible_check = {dict_to_string_leader(best_feasible_action): True}
    
    # Check remaining actions in final population to see if any are better
    for k_leader in leader_actions:
        
        current_fitness = fitness_leader(k_leader)
        if current_fitness > best_feasible_fitness:
            # Fittest action found so far, it is worth checking for feasibility
            
            try:  # Assume we already know whether the action is feasible
                if is_feasible_check[dict_to_string_leader(k_leader)]:
                    # New best action found
                    best_feasible_fitness = current_fitness
                    best_feasible_action = k_leader
            
            except KeyError:  # The action has not yet been checked for feasibility
                corresponding_strategy = find_follower_strategy_MA(k_leader)
                if leader_budget_satisfied_k(
                    k_leader, {s: corresponding_strategy[s] for s in scenarios}
                ):
                    # New best action found
                    is_feasible_check[dict_to_string_leader(k_leader)] = True
                    best_feasible_fitness = current_fitness
                    best_feasible_action = k_leader
                else:
                    # Action is infeasible
                    is_feasible_check[dict_to_string_leader(k_leader)] = False
    
    return best_feasible_action
