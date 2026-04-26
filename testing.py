""" Comparing the algorithms """


import time
import datetime


seed = 1  # Change seed 1-5 to change the runs of the Master Algorithm


from generate_instances import *
if num_vertices_unoccupied == 10:
    from exact_method import *  # Cannot run this script with 60 unoccupied vertices
from master_alg import *
from market_potential_alg import *


# If exact_method.py is not imported, calculate the size of K
if num_vertices_unoccupied != 10:
    len_K = 1
    for v in vertices_unoccupied:
        len_K *= (MAX_QUALITY_INDICES[v] + 1)
    print(f"K contains {len_K} elements")


# MASTER ALGORITHM #####################################################################

# Set seed depending on the run
random.seed(seed)

print(f"\nSolving MA...")
print(f"Solving started at {datetime.datetime.now()}")

# Run the Master Algorithm for one run and time it
time_2 = time.perf_counter()
leader_MA = find_leader_MA()
time_3 = time.perf_counter()

print(f"RUN {seed}")
print(f"MA time: {round(time_3 - time_2)} seconds")

try:  # Assume we can solve instance exactly
    # Calculate adjusted objective function value of the obtained action
    exact_response_to_MA = {s: find_follower_exact(leader_MA, s) for s in scenarios}
    MA_adjusted_ofv = master_objective_fun_k(
        leader_MA, {s: exact_response_to_MA[s] for s in scenarios}
    )
    print(f"MA adjusted OFV: {int(MA_adjusted_ofv)}")
except NameError:  # Instance is too big to solve exactly
    # Calculate predicted objective function value of the obtained action
    MA_predicted_ofv = master_objective_fun_k(
        leader_MA,
        {s: find_follower_strategy_MA(leader_MA)[s] for s in scenarios}
    )
    print(f"MA predicted OFV: {int(MA_predicted_ofv)}")
print("")


# MARKET POTENTIAL ALGORITHM ###########################################################

print("Solving MP...")

# Run the Market Potential Algorithm and time it
time_0 = time.perf_counter()
leader_MP = find_leader_MP()
time_1 = time.perf_counter()

print(f"MP time: {time_1 - time_0} seconds")
try:  # Assume we can solve instance exactly
    # Calculate adjusted objective function value of the obtained action
    exact_response_to_MP = {s: find_follower_exact(leader_MP, s) for s in scenarios}
    MP_adjusted_ofv = master_objective_fun_k(
        leader_MP, {s: exact_response_to_MP[s] for s in scenarios}
    )
    print(f"MP adjusted OFV: {int(MP_adjusted_ofv)}")
except NameError:  # Instance is too big to solve exactly
    # Calculate predicted objective function value of the obtained action
    predicted = {s: find_follower_MP(leader_MP, s) for s in scenarios}
    MP_predicted_ofv = master_objective_fun_k(
        leader_MP, {s: find_follower_MP(leader_MP, s) for s in scenarios}
    )
    print(f"MP predicted OFV: {MP_predicted_ofv}")
print("")


# EXACT METHOD #########################################################################

try:  # Assume we can solve instance exactly
    print(f"Solving exactly...")
    print(f"Solving started at {datetime.datetime.now()}")
    
    # Find optimal solution using exact method and time it
    time_4 = time.perf_counter()
    leader_exact = find_leader_exact()
    time_5 = time.perf_counter()
    
    # Calculate objective function value of the obtained action
    response_exact = {s: find_follower_exact(leader_exact, s) for s in scenarios}
    exact_ofv = master_objective_fun_k(
        leader_exact, {s: response_exact[s] for s in scenarios}
    )
    print(f"Exact time: {round(time_5 - time_4)}")
    print(f"Exact OFV: {exact_ofv}")
except NameError:  # Instance is too big to solve exactly
    print("Too big to solve exactly")
