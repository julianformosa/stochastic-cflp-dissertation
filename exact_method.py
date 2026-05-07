""" Solves the problem instance by exhaustively checking all possible actions """


import itertools
from generate_instances import *


# Calculate size of K and generate it
print("Generating K...")
maximum_ranges = []
for v in vertices_unoccupied:
    maximum_ranges.append(range(MAX_QUALITY_INDICES[v] + 1))

K = [0] * len_K
for i, sequence in enumerate(
    # All distinct permutations of 0, 1, ..., q_v^max for V_U times
    itertools.product(*maximum_ranges)
):
    # Convert the permutation into a dictionary encoding an action
    K[i] = {v: sequence[j] for j, v in enumerate(vertices_unoccupied)}

# Remove actions from K if they have restricted quality indices
K = [
    k
    for k in K
    if all(k[v] <= MAX_QUALITY_INDICES[v] for v in vertices_unoccupied)
]


def K_follower(k_leader: dict, s: int) -> list:
    """
    Removes actions from K if they overlap with Leader's action, build more
    facilities than allowed, or exceed follower's budget under scenario s.
    """
    
    # Create a list of vertices occupied by Leader's newly built facilities
    leader_vertices = [v for v in vertices_unoccupied if k_leader[v] != 0]
    
    feasible_actions = [
        k
        for k in K
        
        # Does not build more facilities than allowed
        if num_facilities_k(k) <= N - num_existing_follower
           
           # Does not build on Leader's facilities
           and all(k[v] == 0 for v in leader_vertices)
           
           # Does not exceed the budget scenario
           and cost_total_k("follower", k_leader, k) <= BUDGETS_FOLLOWER[s]
    ]
    
    return feasible_actions


def find_follower_exact(k_leader: dict, s: int) -> dict:
    """ Find's follower's response when Leader plays k_leader under scenario s """
    
    # If Leader has taken up all unoccupied vertices, then Follower has nowhere to build on
    if all(k_leader[v] != 0 for v in vertices_unoccupied):
        return {v: 0 for v in vertices_unoccupied}
    
    # Reduce the search space to the actions given by K_follower()
    feasible_actions = K_follower(k_leader, s)
    
    # Store result from first action
    best_actions = [feasible_actions[0]]
    best_profit = profit_k(
        "follower", k_leader, best_actions[0]
    )
    
    # Check the remaining actions to see if any are better
    for k in feasible_actions:
        current_profit = profit_k("follower", k_leader, k)
        if current_profit > best_profit:
            best_actions = [k]
            best_profit = current_profit
        elif current_profit == best_profit:
            best_actions.append(k)
    
    if len(best_actions) == 1:  # Follower's best action is unique
        return best_actions[0]
    
    else:  # Follower has multiple actions which give the same profit
        # We choose the action which minimises Leader's profit
        
        # Store result from first action
        best_action = best_actions[0]
        worst_profit = profit_k("leader", k_leader, best_action)
        
        # Check the remaining actions to see if any are better
        for k in best_actions:
            current_profit = profit_k("leader", k_leader, k)
            if current_profit < worst_profit:
                best_action = k
                worst_profit = current_profit
                
        return best_action


def find_leader_exact() -> dict:
    """ Find's Leader's optimal action """
    
    # Reduce the search space to avoid checking actions which build too many facilities
    feasible_actions = [
        k
        for k in K
        if num_facilities_k(k) <= N - num_existing_leader
    ]
    
    # Store result from the first action
    best_action = feasible_actions[0]
    corresponding_response = {
        s: find_follower_exact(best_action, s)
        for s in scenarios
    }
    best_objective = master_objective_fun_k(
        best_action, corresponding_response
    )
    
    # Check remaining actions to see if any are better
    actions_evaluated = 1
    print(f"{len(feasible_actions)} to evaluate")
    for k in feasible_actions[1:]:
        
        # Find Follower's response
        corresponding_response = {s: find_follower_exact(k, s) for s in scenarios}
        
        # Calculate Leader's resulting objective function value
        objective = master_objective_fun_k(k, corresponding_response)
        
        if objective >= best_objective and leader_budget_satisfied_k(k, corresponding_response):
            # Action is feasible and as good as best action found so far
            best_action = k
            best_objective = objective
            print(best_objective, best_action)
        
        actions_evaluated += 1
        # Print progress for debugging purposes
        if actions_evaluated % 1_000 == 0:
            print(f"{actions_evaluated} actions evaluated out of {len(feasible_actions)}")
    
    return best_action
