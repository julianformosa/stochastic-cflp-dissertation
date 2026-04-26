""" Performs parameter tuning for the Master Algorithm """


import time
from datetime import datetime
from exact_method import *
from generate_instances import *
from random import choice
from collections import Counter
import pandas as pd


seed = 20  # Change seed 1-20 to change trial


# SETTING UP MASTER ALGORITHM ##########################################################

# Randomly choose different parameters depending on the trial --------------------------

random.seed(seed)

fitness_function_type = choice([1, 2])

PI_1 = choice([1000, 10_000, 50_000])
PI_2 = choice([1000, 10_000, 50_000])
POP_SIZE = choice([50, 100, 250, 500])
MAX_ITERATIONS = choice([3, 5, 10])
MIN_IDENTICAL_INDIVIDUALS = choice(
    [
        math.floor(POP_SIZE / 2),
        math.floor(2 * POP_SIZE / 3),
        math.floor(3 * POP_SIZE / 4)
    ]
)
SCENARIO_COPY_RATE = choice([1, 5, 10, 20])

ITERATIONS_GA = choice([3, 5, 10])
NUM_PARENT_PAIRS = choice(
    [
        math.floor(POP_SIZE / 5),
        math.floor(POP_SIZE / 4),
        math.floor(POP_SIZE / 3),
        math.floor(POP_SIZE / 2)
    ]
)
MUTATION_PROBABILITY = choice([0.001, 0.01, 0.1])
TOURNAMENT_SIZE = choice([2, 3, 5, 10])
NUM_CHILDREN = choice(range(2, math.floor(POP_SIZE / NUM_PARENT_PAIRS) + 1))
ELITISM_RATE = POP_SIZE - NUM_CHILDREN * NUM_PARENT_PAIRS
SELECTION_METHOD = choice(["truncation", "tournament", "roulette"])

TS_RATE = choice([1, 3, 5, 10])
ITERATIONS_TS = choice([3, 5, 10])
NUM_NEIGHBOURS = choice([3, 5, 10])
TENURE = choice(range(2, math.floor(ITERATIONS_TS / 3) + 2))


# Set up Master Algorithm --------------------------------------------------------------
# See 'master_alg.py' for comments and explanation of code

def stopping_condition(pop: list, iteration: int) -> bool:
    if iteration >= MAX_ITERATIONS:
        return True
    pop_as_strings = [dict_to_string_leader(k) for k in pop]
    counts = Counter(pop_as_strings)
    identical_individuals_count = counts.most_common()[0][1]
    if identical_individuals_count >= MIN_IDENTICAL_INDIVIDUALS:
        return True
    return False
    

fitness_follower_dict = dict()
fitness_leader_dict = dict()


def fitness_follower(k_follower: dict, k_leader: dict, s: int) -> float:
    instance_string = dict_to_string_follower(k_follower, k_leader, s)
    try:
        return fitness_follower_dict[instance_string]
    except KeyError:
        term_1 = profit_k("follower", k_leader, k_follower)
        term_2 = max(
            0, cost_total_k("follower", k_leader, k_follower) - BUDGETS_FOLLOWER[s]
        )
        
        # Use different fitness function depending on the parameters
        if fitness_function_type == 1:
            term_2 = chi_0(term_2)
            
        term_2 *= PI_1
        fitness_follower_dict[instance_string] = term_1 - term_2
        return term_1 - term_2


def fitness_leader(k_leader: dict) -> float:
    instance_string = dict_to_string_leader(k_leader)
    try:
        return fitness_leader_dict[instance_string]
    except KeyError:
        strategy = find_follower_strategy_MA(k_leader)
        term_1 = master_objective_fun_k(
            k_leader, {s: strategy[s] for s in scenarios}
        )
        probability_obeying_budget = sum(
            SCENARIO_PROBABILITIES[s] * chi_k(k_leader, strategy[s])
            for s in scenarios
        )
        term_2 = max([0, RHO - probability_obeying_budget])
        
        # Use different fitness function depending on the parameters
        if fitness_function_type == 1:
            term_2 = chi_0(term_2)
            
        term_2 *= PI_2
        fitness_leader_dict[instance_string] = term_1 - term_2
        return term_1 - term_2


def find_fitness(evaluated_action: dict, player: str, k_leader: dict, s: int) -> float:
    if player == "leader":
        return fitness_leader(evaluated_action)
    else:
        return fitness_follower(evaluated_action, k_leader, s)


def truncation(pop: list, player: str, k_leader: dict, s: int) -> list:
    sorted_pop = sorted(pop, key=lambda l: find_fitness(l, player, k_leader, s))
    parent_pool = sorted_pop[: 2 * NUM_PARENT_PAIRS]
    parents = []
    while len(parents) < NUM_PARENT_PAIRS:
        [parent_1, parent_2] = random.sample(parent_pool, 2)
        parent_pool.remove(parent_1)
        parent_pool.remove(parent_2)
        parents.append([parent_1, parent_2])
    return parents


def roulette(pop: list, player: str, k_leader: dict, s: int) -> list:
    parents = []
    parent_pool = pop.copy()
    weights = [find_fitness(k, player, k_leader, s) for k in parent_pool]
    if any(w < 0 for w in weights):
        most_negative_weight = min(weights)
        weights = [w - most_negative_weight + 1 for w in weights]
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
    parents = []
    parent_pool = pop.copy()
    while len(parents) < NUM_PARENT_PAIRS:
        tournament_list = random.sample(parent_pool, TOURNAMENT_SIZE)
        sorted_tournament = sorted(
            tournament_list, key=lambda l: find_fitness(l, player, k_leader, s)
        )
        parent_pair = sorted_tournament[:2]
        parents.append(parent_pair)
    return parents


def recombination(k_1: dict, k_2: dict) -> dict:
    offspring = {v: 0 for v in vertices_unoccupied}
    vertices_1 = {v for v in vertices_unoccupied if k_1[v] != 0}
    vertices_2 = {v for v in vertices_unoccupied if k_2[v] != 0}
    for v in vertices_1 & vertices_2:
        offspring[v] = random.choice([k_1[v], k_2[v]])
    vertices_available = random.sample(
        list(vertices_1 ^ vertices_2),
        sum(v != 0 for v in vertices_1) - len(vertices_1 & vertices_2)
    )
    for v in vertices_available:
        offspring[v] = max(k_1[v], k_2[v])
    return offspring


def mutation(k: dict, player: str, k_leader) -> dict:
    m_1 = random.random()
    if m_1 < MUTATION_PROBABILITY: 
        feasible = False
        while not feasible:
            m_2 = random.choice([1, 2, 3, 4])
            try:
                if m_2 == 1:
                    v = random.choice([v for v in vertices_unoccupied if k[v] == 0])
                    k[v] = random.choice(range(1, MAX_QUALITY_INDICES[v] + 1))
                elif m_2 == 2:
                    v = random.choice([v for v in vertices_unoccupied if k[v] != 0])
                    k[v] = 0
                elif m_2 == 3:
                    v = random.choice([v for v in vertices_unoccupied if k[v] != 0])
                    k[v] = random.choice(range(1, MAX_QUALITY_INDICES[v] + 1))
                else:  # m_2 == 4
                    v = random.choice(vertices_unoccupied)
                    u = random.choice([u for u in vertices_unoccupied if u != v])
                    k[v], k[u] = k[u], k[v]
                feasible = obeys_building_constraints_k(k, player, k_leader)
            except IndexError:
                feasible = False
    return k


def genetic_algorithm(pop_old: list, player: str, k_leader: dict, s: int) -> list:
    if SELECTION_METHOD == "truncation":
        parents = truncation(pop_old, player, k_leader, s)
    elif SELECTION_METHOD == "roulette":
        parents = roulette(pop_old, player, k_leader, s)
    else:
        parents = tournament(pop_old, player, k_leader, s)
    pop_new = []
    children_created = 0
    while children_created < NUM_CHILDREN:
        for [parent_1, parent_2] in parents:
            offspring = recombination(parent_1, parent_2)
            offspring = mutation(offspring, player, k_leader)
            pop_new.append(offspring)
        children_created += 1
    sorted_pop_old = sorted(
        pop_old, key=lambda l: find_fitness(l, player, k_leader, s), reverse=True
    )
    for k in sorted_pop_old[:ELITISM_RATE]:
        pop_new.append(k)
    return pop_new


def generate_neighbourhood(k_current: dict, player: str, k_leader: dict) -> list:
    neighbours = []
    while len(neighbours) < NUM_NEIGHBOURS:
        k_neighbour = k_current.copy()
        if player == "leader":
            possible_vertices = vertices_unoccupied
        else:
            possible_vertices = [u for u in vertices_unoccupied if k_leader[u] == 0]
        if possible_vertices:
            v_diff = random.choice(possible_vertices)
            possible_quality_indices = [
                q for q in range(MAX_QUALITY_INDICES[v_diff] + 1)
                if k_current[v_diff] != q
            ]
            q_diff = random.choice(possible_quality_indices)
            k_neighbour[v_diff] = q_diff
            if obeys_building_constraints_k(k_neighbour, player, k_leader):
                neighbours.append(k_neighbour)
        else:
            return [k_neighbour]
    return neighbours


def best_from_list(lst: list, player: str, k_leader: dict, s: int) -> dict:
    best_action = lst[0].copy()
    best_fitness = find_fitness(lst[0], player, k_leader, s)
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
    tabu_neighbours = []
    valid_neighbours = []
    for k_neighbour in neighbourhood:
        v_diff = -100
        for v in vertices_unoccupied:
            if k_current[v] != k_neighbour[v]:
                v_diff = v
                break
        if [v_diff, k_neighbour[v_diff]] in tabu_list:
            tabu_neighbours.append(k_neighbour)
        else:
            valid_neighbours.append(k_neighbour)
    try:
        best_valid_action = best_from_list(valid_neighbours, player, k_leader, s)
    except IndexError:  # All neighbours are tabu
        best_tabu_action = best_from_list(tabu_neighbours, player, k_leader, s)
        return best_tabu_action
    try:
        best_tabu_action = best_from_list(tabu_neighbours, player, k_leader, s)
    except IndexError:  # No tabu neighbours
        return best_valid_action
    if f_best < find_fitness(best_tabu_action, player, k_leader, s):
        return best_tabu_action
    else:
        return best_valid_action


def tabu_search(k_old: dict, player: str, k_leader: dict, s: int) -> dict:
    k_current = k_old
    k_best = k_old
    f_best = find_fitness(k_old, player, k_leader, s)
    tabu_list = []
    iteration = 0
    while iteration < ITERATIONS_TS:
        neighbours = generate_neighbourhood(k_current, player, k_leader)
        best_neighbour = select_neighbour(
            neighbours, k_current, f_best, tabu_list, player, k_leader, s
        )
        v_diff = -100
        for v in vertices_unoccupied:
            if k_current[v] != best_neighbour[v]:
                v_diff = v
                break
        tabu_list.append([v_diff, k_current[v_diff]])
        if f_best < find_fitness(best_neighbour, player, k_leader, s):
            f_best = find_fitness(best_neighbour, player, k_leader, s)
            k_best = best_neighbour.copy()
        if iteration > TENURE:
            tabu_list.pop(0)
        k_current = best_neighbour
        iteration += 1
    return k_best


def generate_follower(k_leader: dict) -> dict:
    k_follower = {v: 0 for v in vertices_unoccupied}
    available_vertices = [v for v in vertices_unoccupied if k_leader[v] == 0]
    tau_f = min(
        N - num_existing_follower,
        num_vertices_unoccupied - num_facilities_k(k_leader)
    )
    selected_vertices = random.sample(available_vertices, tau_f)
    for v in selected_vertices:
        q = random.choice(list(range(1, MAX_QUALITY_INDICES[v] + 1)))
        k_follower[v] = q
    return k_follower


follower_strategy_table = dict()


def find_follower_strategy_MA(k_leader: dict) -> dict:
    try:
        return follower_strategy_table[dict_to_string_leader(k_leader)]
    except KeyError:
        if all(k_leader[v] != 0 for v in vertices_unoccupied):
            return {s: {v: 0 for v in vertices_unoccupied} for s in scenarios}
        follower_actions = []
        strategy = dict()
        for s in scenarios:
            while len(follower_actions) < POP_SIZE:
                k_follower = generate_follower(k_leader)
                follower_actions.append(k_follower)
            iteration = 0
            while not stopping_condition(follower_actions, iteration):
                for _ in range(ITERATIONS_GA):
                    follower_actions = genetic_algorithm(
                        follower_actions, "follower", k_leader, s
                    )
                follower_actions = list(
                    sorted(
                        follower_actions,
                        key=lambda l: fitness_follower(l, k_leader, s),
                        reverse=True
                    )
                )
                for _ in range(TS_RATE):
                    action = follower_actions[0].copy()
                    follower_actions.pop(0)
                    improved_action = tabu_search(action, "follower", k_leader, s)
                    follower_actions.append(improved_action)
                iteration += 1
            best_feasible_action = {v: 0 for v in vertices_unoccupied}
            best_feasible_fitness = 0
            is_feasible_check = {
                dict_to_string_follower(best_feasible_action, k_leader, s): True
            }
            for k in follower_actions:
                current_fitness = fitness_follower(k, k_leader, s)
                if current_fitness > best_feasible_fitness:
                    try:
                        if is_feasible_check[dict_to_string_follower(k, k_leader, s)]:
                            best_feasible_fitness = current_fitness
                            best_feasible_action = k
                    except KeyError:
                        if cost_total_k("follower", k_leader, k) <= BUDGETS_FOLLOWER[s]:
                            best_feasible_fitness = current_fitness
                            best_feasible_action = k
                            is_feasible_check[dict_to_string_follower(k, k_leader, s)] = True
                        else:
                            is_feasible_check[dict_to_string_follower(k, k_leader, s)] = False
            strategy[s] = best_feasible_action
            sorted_follower_actions = list(
                sorted(
                    follower_actions,
                    key=lambda l: fitness_follower(l, k_leader, s),
                    reverse=True
                )
            )
            follower_actions = sorted_follower_actions[:SCENARIO_COPY_RATE]
        follower_strategy_table[dict_to_string_leader(k_leader)] = strategy
        return strategy


def generate_leader() -> dict:
    k = {v: 0 for v in vertices_unoccupied}
    tau_l = min(N - num_existing_leader, num_vertices_unoccupied)
    selected_vertices = random.sample(vertices_unoccupied, tau_l)
    for v in selected_vertices:
        q = random.choice(range(1, MAX_QUALITY_INDICES[v] + 1))
        k[v] = q
    return k


def find_leader_MA() -> dict:
    leader_actions = []
    while len(leader_actions) < POP_SIZE:
        k_leader = generate_leader()
        leader_actions.append(k_leader)
    iteration = 0
    while not stopping_condition(leader_actions, iteration):
        print(f"Leader iteration {iteration}")
        for i in range(ITERATIONS_GA):
            print(f"GA iteration {i}")
            leader_actions = genetic_algorithm(leader_actions, "leader", None, None)
        leader_actions = list(
            sorted(
                leader_actions,
                key=lambda l: fitness_leader(l),
                reverse=True
            )
        )
        for j in range(TS_RATE):
            print(f"TS improvement {j}")
            action = leader_actions[0].copy()
            leader_actions.pop(0)
            improved_action = tabu_search(action, "leader", None, None)
            leader_actions.append(improved_action)
        iteration += 1
    best_feasible_action = {v: 0 for v in vertices_unoccupied}  # Certainly feasible
    best_feasible_fitness = 0
    is_feasible_check = {dict_to_string_leader(best_feasible_action): True}
    for k_leader in leader_actions:
        current_fitness = fitness_leader(k_leader)
        if current_fitness > best_feasible_fitness:
            try:
                if is_feasible_check[dict_to_string_leader(k_leader)]:
                    best_feasible_fitness = current_fitness
                    best_feasible_action = k_leader
            except KeyError:
                corresponding_strategy = find_follower_strategy_MA(k_leader)
                if leader_budget_satisfied_k(
                    k_leader, {s: corresponding_strategy[s] for s in scenarios}
                ):
                    is_feasible_check[dict_to_string_leader(k_leader)] = True
                    best_feasible_fitness = current_fitness
                    best_feasible_action = k_leader
                else:
                    is_feasible_check[dict_to_string_leader(k_leader)] = False
    return best_feasible_action


# PARAMETER TUNING #####################################################################

# Solving exactly only needs to be done once, so that the exact solving time can be found
# print("")
# print(f"Solving exactly...")
# exact_time_start = time.perf_counter()
# leader_exact = find_leader_exact()
# exact_time_end = time.perf_counter()
# exact_response = {s: find_follower_exact(leader_exact, s) for s in scenarios}
# exact_ofv = master_objective_fun_k(
#     leader_exact, {s: exact_response[s] for s in scenarios}
# )
# print("")
# print(f"Exact optimal ofv: {exact_ofv}")
# print(f"Exact time taken: {round(exact_time_end - exact_time_start)}")

NUM_RUNS = 5  # Number of times the Master Algorithm is run for each configuration
tuning_start = time.perf_counter()
parameters = [
    fitness_function_type,
    PI_1,
    PI_2,
    POP_SIZE,
    MAX_ITERATIONS,
    MIN_IDENTICAL_INDIVIDUALS,
    SCENARIO_COPY_RATE,
    ITERATIONS_GA,
    SELECTION_METHOD,
    NUM_PARENT_PAIRS,
    MUTATION_PROBABILITY,
    NUM_CHILDREN,
    TOURNAMENT_SIZE,
    TS_RATE,
    ITERATIONS_TS,
    NUM_NEIGHBOURS,
    TENURE
]
print("")
print(parameters)
parameters_df = pd.DataFrame(parameters)

# Outputting parameter values to Excel
with pd.ExcelWriter(
    "Parameter tuning.xlsx", engine="openpyxl", mode="a", if_sheet_exists="overlay"
) as writer:
    parameters_df.to_excel(
        writer,
        sheet_name="Sheet5",
        index=False,
        header=False,
        startrow=1,
        startcol=seed
    )

# Performing the individual runs
for j in range(NUM_RUNS):
    print("")
    print(datetime.now().time())
    print(f"RUN {j + 1}")
    
    # Reset stored fitnesses and strategies from previous runs
    fitness_follower_dict = dict()
    fitness_leader_dict = dict()
    follower_strategy_table = dict()
    
    # Perform one run of the Master Algorithm and time it
    start = time.time()
    leader_action_out = find_leader_MA()
    end = time.time()
    
    # Find Follower's corresponding response
    exact_response = {s: find_follower_exact(leader_action_out, s) for s in scenarios}
    
    # Calculate the adjusted objective function value
    adjusted_ma_ofv = master_objective_fun_k(
        leader_action_out, {s: exact_response[s] for s in scenarios}
    )
    
    # Store the results
    results = [j + 1, int(adjusted_ma_ofv), round(end - start)]
    print(results)

tuning_end = time.perf_counter()
    
print("")
print(f"Time taken for tuning: {round(tuning_end - tuning_start)}")
