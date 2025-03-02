import numpy as np
import networkx as nx
import itertools
from collections import defaultdict

class BayesianNetwork:
    def __init__(self, adjacency_matrix, nodes):
        self.graph = nx.DiGraph(adjacency_matrix)
        self.nodes = nodes

    def d_separation(self, X, Y, Z):
        """Check if X and Y are d-separated given Z."""
        if isinstance(X, str):
            X = self.nodes.index(X)
        if isinstance(Y, str):
            Y = self.nodes.index(Y)
        Z = [self.nodes.index(z) for z in Z]  # Convert all Z variables to indices

        moralized_graph = nx.moral_graph(self.graph)
        moralized_graph.remove_nodes_from(Z)
        return not nx.has_path(moralized_graph, X, Y)

    def variable_elimination(self, factors, query_var, evidence):
        """Perform exact inference using variable elimination."""
        relevant_factors = [f for f in factors if query_var in f['variables'] or any(e in f['variables'] for e in evidence)]
        for var in set(itertools.chain(*[f['variables'] for f in relevant_factors])) - set([query_var]) - set(evidence.keys()):
            relevant_factors = self.eliminate_variable(relevant_factors, var)
        return self.normalize_factors(relevant_factors, query_var)
    
    def eliminate_variable(self, factors, var):
        """Sum out a variable from all relevant factors."""
        relevant = [f for f in factors if var in f['variables']]
        new_factors = [f for f in factors if var not in f['variables']]
        combined_factor = self.multiply_factors(relevant)
        summed_out = self.sum_out_variable(combined_factor, var)
        new_factors.append(summed_out)
        return new_factors
    
    def multiply_factors(self, factors):
        """Multiply factors using NumPy arrays."""
        result = factors[0]['values']
        for factor in factors[1:]:
            result = np.multiply(result, factor['values'])
        return {'variables': factors[0]['variables'], 'values': result}
    
    def sum_out_variable(self, factor, var):
        """Sum out a variable using NumPy arrays."""
        axis = factor['variables'].index(var)
        summed_values = np.sum(factor['values'], axis=axis)
        new_variables = [v for v in factor['variables'] if v != var]
        return {'variables': new_variables, 'values': summed_values}
    
    def normalize_factors(self, factors, query_var):
        """Normalize the probability distribution using NumPy."""
        total = np.sum(factors[0]['values'])
        factors[0]['values'] /= total
        return factors[0]
    
    def gibbs_sampling(self, query_var, evidence, num_samples=10000, burn_in=1000):
        """Perform approximate inference using Gibbs sampling."""
        samples = []
        # Initialize random values for all variables
        state = {var: np.random.choice([0, 1]) for var in self.nodes}
        # Fix evidence values
        for var, val in evidence.items():
            state[var] = val
        
        for i in range(num_samples + burn_in):
            for var in self.nodes:
                if var not in evidence:
                    state[var] = self.sample_var(var, state)
            if i >= burn_in:
                samples.append(state[query_var])
        
        return sum(samples) / len(samples)

    def sample_var(self, var, state):
        """Sample a variable using its Markov blanket."""
        # Convert the variable name to index
        var_idx = self.nodes.index(var)
        
        # Get the Markov blanket (parents & children)
        blanket = set(self.graph.predecessors(var_idx)) | set(self.graph.successors(var_idx))  # Parents & children
        relevant_factors = [f for f in self.factors if any(n in f['variables'] for n in blanket)]
        
        prob_dist = np.ones(2)
        for factor in relevant_factors:
            index = tuple(state[v] for v in factor['variables'])
            prob_dist *= factor['values'][index]
        
        prob_dist /= np.sum(prob_dist)  # Normalize
        return np.random.choice([0, 1], p=prob_dist)

