"""Solves the Master Program using the Market Potential Algorithm"""


from generate_instances import *


def market_potential(u: int) -> float:
    """ The market potential of vertex u """
    return MAX_DEMANDS[u] * QUALITY_LEVELS[MAX_QUALITY_INDICES[u]]


def find_follower_MP(k_leader: dict, s: int) -> dict:
    """ Finds Follower's response to k_leader under scenario s """
    
    # Initialise a zero vector
    k_follower = {v: 0 for v in vertices_unoccupied}
    
    # Find the vertices not used by leader
    vertices_available = [v for v in vertices_unoccupied if k_leader[v] == 0]
    
    # Sort vertices in decreasing order of market potential
    vertices_available = sorted(
        vertices_available, key=lambda x: market_potential(x), reverse=True
    )
    
    for v in vertices_available:
        if num_facilities_k(k_follower) < N - num_existing_follower:
            # Follower is allowed to build more facilities
            
            # Let Follower build a facility with the highest possible quality level
            q = MAX_QUALITY_INDICES[v]
            k_follower[v] = q
            
            # Calculate cost of Follower's action
            current_cost = cost_total_k("follower", k_leader, k_follower)
            
            
            while current_cost > BUDGETS_FOLLOWER[s]:
                # Follower's action is infeasible
                
                # Decrease the quality of the facility by one level
                q = q - 1
                k_follower[v] = q
                current_cost = cost_total_k("follower", k_leader, k_follower)
    
    return k_follower


def find_leader_MP():
    """ Finds Leader's best action """
    
    # Initialise a zero vector
    k_leader = {v: 0 for v in vertices_unoccupied}
    
    # Sort unoccupied vertices in decreasing order of market potential
    vertices_available = sorted(
        vertices_unoccupied, key=lambda x: market_potential(x), reverse=True
    )
    
    for v in vertices_available:
        if num_facilities_k(k_leader) < N - num_existing_leader:
            # Leader is allowed to build more facilities
            
            # Let Leader build a facility with the highest possible quality level
            q = MAX_QUALITY_INDICES[v]
            k_leader[v] = q
            
            # Find Follower's corresponding strategy
            follower_strategy = {s: find_follower_MP(k_leader, s) for s in scenarios}
            
            while not leader_budget_satisfied_k(k_leader, follower_strategy):
                # Leader's budget constraint not satisfied
                
                # Decrease the quality of the facility by one level
                q = q - 1
                k_leader[v] = q
    
    return k_leader
