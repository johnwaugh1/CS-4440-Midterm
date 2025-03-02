import numpy as np
from bayesian import BayesianNetwork

# Define the adjacency matrix
adjacency_matrix = np.array([
    [0, 1, 1, 0, 0],  # Burglary influences Alarm
    [0, 0, 1, 0, 0],  # Earthquake influences Alarm
    [0, 0, 0, 1, 1],  # Alarm influences JohnCalls & MaryCalls
    [0, 0, 0, 0, 0],  # JohnCalls has no children
    [0, 0, 0, 0, 0]   # MaryCalls has no children
])
nodes = ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls']

# Define Conditional Probability Tables (CPTs) as factors
factors = [
    {'variables': ['Burglary'], 'values': np.array([0.999, 0.001])},
    {'variables': ['Earthquake'], 'values': np.array([0.998, 0.002])},
    {'variables': ['Burglary', 'Earthquake', 'Alarm'],
     'values': np.array([
         [[0.999, 0.001], [0.71, 0.29]],  # Burglary=0, Earthquake=0/1
         [[0.06, 0.94], [0.05, 0.95]]     # Burglary=1, Earthquake=0/1
     ])},
    {'variables': ['Alarm', 'JohnCalls'],
     'values': np.array([[0.95, 0.05], [0.1, 0.9]])},
    {'variables': ['Alarm', 'MaryCalls'],
     'values': np.array([[0.99, 0.01], [0.3, 0.7]])}
]

# Initialize the Bayesian Network
bn = BayesianNetwork(adjacency_matrix, nodes)
bn.factors = factors  # Assign the defined factors

# Perform D-Separation Check
print("D-Separation (Burglary ⊥ MaryCalls | Alarm):", bn.d_separation('Burglary', 'MaryCalls', ['Alarm']))
print("D-Separation (Burglary ⊥ JohnCalls | Alarm):", bn.d_separation('Burglary', 'JohnCalls', ['Alarm']))
print("D-Separation (Burglary ⊥ Earthquake | Alarm):", bn.d_separation('Burglary', 'Earthquake', ['Alarm']))

# Perform Exact Inference
query_var = 'Alarm'
evidence = {'JohnCalls': 1, 'MaryCalls': 1}
print("Exact Inference P(Alarm | JohnCalls=1, MaryCalls=1):", bn.variable_elimination(factors, query_var, evidence))

# Perform Approximate Inference via Gibbs Sampling
print("Gibbs Sampling P(Alarm | JohnCalls=1, MaryCalls=1):", bn.gibbs_sampling(query_var, evidence))

query_var = 'Burglary'
evidence = {'JohnCalls': 1, 'MaryCalls': 1}
print("Gibbs Sampling P(Burglary | JohnCalls=1, MaryCalls=1):", bn.gibbs_sampling(query_var, evidence))

