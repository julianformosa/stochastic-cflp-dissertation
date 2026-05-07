""" Plots the solutions to the problem instances solved in Chapter 5 """

import osmnx as ox
import random


num_vertices_unoccupied = 60      # Either 10, 30 or 60
random.seed(80)  # Either 5, 25, 60, 80 or 90

# The solutions found for one scenario, elements of K
leader_solution =  {4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 2, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 2, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 4, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0}
follower_solution = {4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 4, 27: 2, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 4, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0}


# PLAYING FIELD ########################################################################

# Get Sliema's street network as a graph
print("Downloading data...")
G = ox.graph.graph_from_place(
    query="Sliema, Malta", network_type="drive", retain_all=False, simplify=True
)

# Ensure G is strongly connected (i.e. any vertex can be reached from any other vertex)
G = ox.truncate.largest_component(G, strongly=True)

# Select nodes for Follower's existing chargers
longitudes = [14.505842, 14.509751, 14.4990198644346]
latitudes = [35.908752, 35.910208, 35.9076972438659]
nodes_occupied = []
for i in range(3):
    # Add the closest node to the coordinates to nodes_occupied
    nodes_occupied.append(ox.distance.nearest_nodes(G, X=longitudes[i], Y=latitudes[i]))

# By calling random.random() a different number of times depending on the size, we
# ensure that every combination of seed + size uses different nodes as the unoccupied vertices.
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
    [3, 3]  # Charger on vertex 3 with quality level 3
]

num_vertices_occupied = len(EXISTING_FOLLOWER)
num_existing_follower = num_vertices_occupied
num_vertices = num_vertices_occupied + num_vertices_unoccupied

# Vertices 1, ..., V_F are occupied
vertices_occupied = list(range(1, num_vertices_occupied + 1))

# Vertices V_F + 1, ..., V_U are unoccupied
vertices_unoccupied = list(range(num_vertices_occupied + 1, num_vertices + 1))

# vertices = [1, 2, ..., V]
vertices = list(range(1, num_vertices + 1))

# Create a dictionary to store the graph's node index for all of our vertices
vertex_to_node = dict()
for v in vertices_occupied:
    vertex_to_node[v] = nodes_occupied[v - 1]
for v in vertices_unoccupied:
    vertex_to_node[v] = nodes_unoccupied[v - num_vertices_occupied - 1]
    

# PLOTTING #############################################################################

# Partition nodes by player and associate them with the quality level of the facility
nodes_occupied = {vertex_to_node[v]: EXISTING_FOLLOWER[v - 1][1] for v in vertices_occupied}
nodes_unoccupied = [vertex_to_node[v] for v in vertices_unoccupied]
nodes_leader = {
    vertex_to_node[v]: leader_solution[v]
    for v in leader_solution
    if leader_solution[v]
}
nodes_follower = {
    vertex_to_node[v]: follower_solution[v]
    for v in follower_solution
    if follower_solution[v]
}

# Choose node size and colour
node_colours = []
node_sizes = []
node_edge_colours = []
for node in G.nodes():
    if node in nodes_occupied:
        node_colours.append("orange")
        node_sizes.append(80 * nodes_occupied[node])
        node_edge_colours.append((0, 0, 0))
    elif node in nodes_leader:
        node_colours.append((0, 0.5, 1))
        node_sizes.append(80 * nodes_leader[node])
        node_edge_colours.append((0, 0, 0))
    elif node in nodes_follower:
        node_colours.append((1, 0, 0))
        node_sizes.append(80 * nodes_follower[node])
        node_edge_colours.append((0, 0, 0))
    elif node in nodes_unoccupied:
        node_colours.append((1, 1, 1))
        node_sizes.append(80)
        node_edge_colours.append((0, 0, 0))
    else:
        node_colours.append((0, 0, 0, 0.1))
        node_sizes.append(10)
        node_edge_colours.append((0, 0, 0, 0.1))

# Plot the network
fig, ax = ox.plot.plot_graph(
    G,
    bgcolor='white',
    node_color=node_colours,
    node_size=node_sizes,
    edge_color=(0, 0, 0, 0.1),
    node_edgecolor=node_edge_colours,
    edge_linewidth=1.5,
    show=True,
    close=False
)
