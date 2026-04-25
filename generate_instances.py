""" Generates problem instances based on the EV charging station industry in Sliema """


import osmnx as ox
import random
import networkx as nx
from functools import cache
import math


num_vertices_unoccupied = 60  # Choose number of unoccupied vertices
random.seed(5)  # Use seed 22 for parameter tuning, and seeds 1-5 for tests


# PLAYING FIELD ########################################################################

# Get a graph of Sliema's street network
print("Downloading data...")
G = ox.graph.graph_from_place(
    query="Sliema, Malta", network_type="drive", retain_all=False, simplify=True
)

# Ensure G is strongly connected (i.e. any vertex can be reached from any other vertex)
G = ox.truncate.largest_component(G, strongly=True)

# Plot graph of the road network
# fig, ax = ox.plot.plot_graph(
#     G,
#     bgcolor='white',
#     node_color='black',
#     node_size=10,
#     edge_color='black',
#     edge_linewidth=1.5,
#     show=True,
#     close=False
# )

# Select nodes for Follower's existing chargers
longitudes = [14.505842, 14.509751, 14.4990198644346]  # Latitudes of existing chargers
latitudes = [35.908752, 35.910208, 35.9076972438659]  # Longitudes of existing chargers
nodes_occupied = []
for i in range(3):
    # Add to nodes_occupied the closest node to the existing charger's coordinates
    nodes_occupied.append(ox.distance.nearest_nodes(G, X=longitudes[i], Y=latitudes[i]))

# Make sure that each example uses different nodes for the unoccupied vertices, even
# when using the same seed, by calling random.random() a different number of times.
for _ in range(num_vertices_unoccupied):
    random.random()

potential_nodes = list(G.nodes)  # The list of all nodes in the network

# Randomly choose some nodes to be the unoccupied vertices
nodes_unoccupied = random.choices(potential_nodes, k=num_vertices_unoccupied)

# Leader has no existing facilities since they are new to the market
EXISTING_LEADER = []
num_existing_leader = 0

# Follower has three existing chargers --- two of quality level 1, and one of quality level 3
EXISTING_FOLLOWER = [
    [1, 1],  # Charger on vertex 1 with quality level 1
    [2, 1],  # Charger on vertex 2 with quality level 1
    [3, 3]   # Charger on vertex 3 with quality level 3
]

num_vertices_occupied = len(EXISTING_FOLLOWER)
num_existing_follower = num_vertices_occupied
num_vertices = num_vertices_occupied + num_vertices_unoccupied

# Vertices 1, ..., V_F are occupied
vertices_occupied = list(range(1, num_vertices_occupied + 1))

# Vertices V_F + 1, ..., V_U are unoccupied
vertices_unoccupied = list(range(num_vertices_occupied + 1, num_vertices + 1))

vertices = list(range(1, num_vertices + 1))

# Create a dictionary to store the graph's node index for all of our vertices
vertex_to_node = dict()
for v in vertices_occupied:
    vertex_to_node[v] = nodes_occupied[v - 1]
for v in vertices_unoccupied:
    vertex_to_node[v] = nodes_unoccupied[v - num_vertices_occupied - 1]

# Initialise a dictionary to store the distances between different vertices.
# This will let us avoid having to recalculate them multiple times
distances = dict()


def distance(start_vertex: int, end_vertex: int) -> float:
    """ Calculates the distance between a pair of vertices, in km """
    
    # Check if the distance is already known by looking in the dictionary we created
    if (start_vertex, end_vertex) in distances.keys():
        return distances[(start_vertex, end_vertex)]
    
    if start_vertex == end_vertex:
        distances[(start_vertex, end_vertex)] = 0
    else:
        start_node = vertex_to_node[start_vertex]
        end_node = vertex_to_node[end_vertex]
        dist = nx.shortest_path_length(G, start_node, end_node, weight="length")
        dist = dist / 1000  # Change from metres to kilometres
        distances[(start_vertex, end_vertex)] = dist  # Store result in the dictionary
    
    return distances[(start_vertex, end_vertex)]


quality_indices = list(range(1, 5))
num_quality_indices = 4

# Create a dictionary to link each vertex with its maximum quality index
MAX_QUALITY_INDICES = dict()
for v in vertices_unoccupied:
    # Randomly pick a quality index as the maximum quality index for this vertex
    MAX_QUALITY_INDICES[v] = random.choices(
        quality_indices, weights=[0.25, 0.25, 0.25, 0.25]
    )[0]

# Create the list of restricted quality-location pairs
restricted_pairs = []
for v in vertices_unoccupied:
    # All pairs (v, q_v^max + 1), ..., (v, Q) are restricted
    for quality in range(MAX_QUALITY_INDICES[v] + 1, num_quality_indices + 1):
        restricted_pairs.append([v, quality])


# OTHER PARAMETERS AND FUNCTIONS #######################################################

scenarios = [1, 2, 3, 4]
num_scenarios = 4
QUALITY_LEVELS = {1: 20, 2: 30, 3: 200, 4: 250}
BUDGET_LEADER = 80_000
BUDGETS_FOLLOWER = {1: 40_000, 2: 80_000, 3: 120_000, 4: 160_000}
SCENARIO_PROBABILITIES = {1: 0.3, 2: 0.4, 3: 0.2, 4: 0.1}
MAX_DEMANDS = {v: math.ceil(816 / num_vertices) for v in vertices}
N = 6
ALPHA = 1
BETA = 2
ETA = 0.0030
MU = {1: 5_000, 2: 10_000, 3: 50_000, 4: 55_000}
EPSILON = 10 ** (-6)
GAMMA = 466
PHI = 1.25
THETA = 2
RHO = 0.95


# HOW WE ENCODE ACTIONS AND STRATEGIES -------------------------------------------------
# The player action (1, 4, 2) in K is encoded using the dictionary {1: 1, 2: 4, 3: 2}.
# A Follower strategy is encoded as a dictionary with keys 1, ..., S representing the
# scenarios, and each key corresponds to the action Follower would play under that scenario.
# --------------------------------------------------------------------------------------


def chi_0(n: float) -> int:
    if n == 0:
        return 0
    else:
        return 1
    

# Attractions
@cache  # Keep attractions in cache since they are used often
def attraction(u: int, v: int, q: int) -> float:
    """ The attraction felt by vertex u to a facility on vertex v with quality index q"""
    return QUALITY_LEVELS[q] / (ALPHA + distance(u, v)) ** BETA


# Total attraction
def total_attraction_k(u: int, kl: dict, kf: dict) -> float:
    """
    The total attraction felt by vertex u towards all facilities, when Leader plays kl
    and Follower plays kf
    """
    
    # Total attraction felt by vertex u towards all of Leader's existing facilities
    existing_attraction_leader = sum(attraction(u, v, q) for [v, q] in EXISTING_LEADER)
    
    # Total attraction felt by vertex u towards all of Follower's existing facilities
    existing_attraction_follower = sum(attraction(u, v, q) for [v, q] in EXISTING_FOLLOWER)
    
    # Total attraction felt by vertex u towards all of Leader's new facilities
    new_attraction_leader_terms = [
        attraction(u, v, kl[v])
        for v in vertices_unoccupied
        if kl[v] != 0
    ]
    new_attraction_leader = sum(new_attraction_leader_terms)
    
    # Total attraction felt by vertex u towards all of Follower's new facilities
    new_attraction_follower_terms = [
        attraction(u, v, kf[v])
        for v in vertices_unoccupied
        if kf[v] != 0
    ]
    new_attraction_follower = sum(new_attraction_follower_terms)
    
    # Return sum of all attractions
    return (
        existing_attraction_leader
        + existing_attraction_follower
        + new_attraction_leader
        + new_attraction_follower
    )


def generated_demand_k(u: int, kl: dict, kf: dict) -> float:
    """
    The level of demand generated at vertex u when Leader plays kl and Follower plays kf
    """
    return MAX_DEMANDS[u] * (1 - math.exp(- ETA * total_attraction_k(u, kl, kf)))


def market_share_k(player: str, kl: dict, kf: dict) -> float:
    """ The player's market share when Leader plays kl and Follower plays kf. """
    
    try:  # Assume total attraction is not zero (i.e. some facilities exist)
        
        # Use different facilities and action depending on the player
        if player == "leader":
            existing_facilities = EXISTING_LEADER
            action = kl
        else:  # player == follower
            existing_facilities = EXISTING_FOLLOWER
            action = kf
        
        # Market share captured by existing facilities
        existing_share = sum(
            [
                (
                    generated_demand_k(u, kl, kf)
                    * attraction(u, v, q)
                    / total_attraction_k(u, kl, kf)
                )
                for [v, q] in existing_facilities
                for u in vertices
            ]
        )
        
        # Market share captured by newly built facilities
        new_share = sum(
            [
                (
                    generated_demand_k(u, kl, kf)
                    * attraction(u, v, action[v])
                    / total_attraction_k(u, kl, kf)
                )
                for v in vertices_unoccupied
                for u in vertices
                if action[v] != 0
            ]
        )
        
        return existing_share + new_share
    
    except ZeroDivisionError:  # Total attraction is zero (i.e. no facilities exist)
        return 0


def operating_cost_k(v: int, kl: dict, kf: dict) -> float:
    """ The location cost for vertex v when Leader plays kl and Follower plays kf """
    return sum(
        [
            generated_demand_k(u, kl, kf) / (PHI + distance(u, v) ** THETA)
            for u in vertices
        ]
    )


def cost_total_k(player: str, kl: dict, kf: dict) -> float:
    """ The player's total cost """
    
    # Use a different action depending on the player
    if player == "leader":
        action = kl
    else:  # Player == follower
        action = kf
    
    total_cost = sum(
        [
            operating_cost_k(v, kl, kf) + MU[action[v]]
            for v in vertices_unoccupied
            if action[v] != 0
        ]
    )
    
    return total_cost


def profit_k(player: str, kl: dict, kf: dict) -> float:
    """ The player's profit when Leader plays kl and Follower plays kf  """
    return GAMMA * market_share_k(player, kl, kf) - cost_total_k(player, kl, kf)


def master_objective_fun_k(kl: dict, follower_strategy_k: dict) -> float:
    """
    Calculates the Master Program objective function value when Leader plays kl and
    Follower uses strategy follower_strategy_k
    """
    
    term_1 = sum(
        [
            SCENARIO_PROBABILITIES[s] * profit_k("leader", kl, follower_strategy_k[s])
            for s in scenarios
        ]
    )
    term_2 = sum(
        [
            SCENARIO_PROBABILITIES[s] * profit_k("leader", kl, follower_strategy_k[s]) ** 2
            for s in scenarios
        ]
    )
    term_3 = sum(
        [
            (SCENARIO_PROBABILITIES[s] * profit_k("leader", kl, follower_strategy_k[s])) ** 2
            for s in scenarios
        ]
    )
    return term_1 - EPSILON * term_2 + EPSILON * term_3


def chi_k(kl: dict, kf: dict) -> int:
    """
    Returns 1 if Leader's budget is respected when Leader plays kl and Follower plays kf
    """
    if cost_total_k("leader", kl, kf) <= BUDGET_LEADER:
        return 1
    else:
        return 0


def leader_budget_satisfied_k(kl: dict, strategy_k: dict) -> bool:
    """
    Checks whether Leader's budget constraint is respected when Leader plays kl and
    Follower uses strategy strategy_k
    """
    terms = [SCENARIO_PROBABILITIES[s] * chi_k(kl, strategy_k[s]) for s in scenarios]
    if sum(terms) >= RHO:
        return True
    else:
        return False


def num_facilities_k(k: dict) -> int:
    """ Counts the number of facilities built by the action k """
    return sum(k[v] != 0 for v in vertices_unoccupied)


def obeys_building_constraints_k(k: dict, player: str, k_leader: dict) -> bool:
    """
    Checks if the building constraints are satisfied (i.e. that the player does not own
    more than the maximum number of facilities, no facility is built with a restricted
    quality level, and Follower does not build on vertices occupied by Leader's newly
    built facilities) for action k.

    k is the player's action.
    If the player is Follower, then k_leader is Leader's action.
    """
    
    # Check if more facilities are owned than the maximum threshold
    if (
        (player == "leader" and num_facilities_k(k) + num_existing_leader > N)
        or (player == "follower" and num_facilities_k(k) + num_existing_follower > N)
    ):
        too_many_facilities = True
    else:
        too_many_facilities = False
    
    # Check if any restricted location-quality pairs are built
    restricted_lq_pairs = False
    for v in vertices_unoccupied:
        if k[v] > MAX_QUALITY_INDICES[v]:
            restricted_lq_pairs = True
    
    # Checks if Follower built a facility on a vertex occupied by one of Leader's newly
    # built facilities
    overlapping_facilities = False
    if player == "follower":
        if [
            v for v in vertices_unoccupied if k[v] != 0 and k_leader[v] != 0
        ]:
            overlapping_facilities = True
    
    
    if too_many_facilities or restricted_lq_pairs or overlapping_facilities:
        # At least one building constraint has been violated
        return False
    else:
        # All building constraints are obeyed
        return True


def dict_to_string_follower(k_follower: dict, k_leader: dict, s: int) -> str:
    """ Converts outcomes into a unique string """
    current_string = ""
    for v in vertices_unoccupied:
        current_string = current_string + str(k_follower[v]) + str(k_leader[v])
    current_string = current_string + str(s)
    return current_string


def dict_to_string_leader(k_leader: dict) -> str:
    """ Converts Leader's action into a string """
    current_string = ""
    for v in vertices_unoccupied:
        current_string = current_string + str(k_leader[v])
    return current_string


# Calculate distances between all pairs of vertices.
print("Finding distances...")
for u in vertices:
    for v in vertices:
        distances[(u, v)] = distance(u, v)
print("...Distances found")
print("")
