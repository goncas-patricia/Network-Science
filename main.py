import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from itertools import count
import sympy as sp
import random

SEED = 7
INITIAL_DEFECTOR_PROB = 0.6
random.seed(SEED)

# Total number of nodes
N_values = list(range(10, 100))  
# Control the scale-free properties
E = 5 
# Initial endowment
b = 2
# Contribution (fraction of the endowment)
c = .1*b
# Risk
r = [0.00, 0.25, 0.50, 0.75, 1.00]
# Models
models = ['H','SH','SG','PD']

def set_all_node_attributes(G, attr_name, attr_value):
    for node in G.nodes():
        G.nodes[node][attr_name] = attr_value

def set_all_edge_attributes(G, attr_name, attr_value):
    for edge in G.edges():
        G.edges[edge][attr_name] = attr_value

def set_node_bool_attribute_with_prob_k(G, attr_name, prob):
    for node in G.nodes():
        if random.random() <= prob:
            G.nodes[node][attr_name] = True
        else:
            G.nodes[node][attr_name] = False

def set_behavior_node_attributes(G, attr_name, cooperator, defector):
    for node in G.nodes():
        if G.nodes[node]['cooperator'] == 1:
            G.nodes[node][attr_name] = cooperator
        else:
            G.nodes[node][attr_name] = defector

def cooperators(G):
    cooperators = [node for node in G.nodes() if G.nodes[node]['cooperator'] == 1]
    return cooperators

def x(G):
    x = len([node for node in G.nodes() if G.nodes[node]['cooperator'] == 1])/len(G.nodes())
    return x

def risk_loss(G, M):
    if len([node for node in G.nodes() if G.nodes[node]['cooperator'] == 1]) < M*len(G.nodes()):
        for node in G.nodes():
            if random.random() <= r[2]:
                G.nodes[node]['endowment'] = 0

def gradient_of_selection(G, model):
        return x * (1 - x) * fitness(G,model)[2]

def fitness(G, model):
    # Cost-Benefit Values (para já, valores estáticos)
    if model == 'H':
        R = b
        T = b-c
        S = b-c
        P = 0
    if model == 'SH':
        R = b
        T = c
        S = -c
        P = 0
    if model == 'SG':
        R = 1
        T = b+c
        S = c
        P = 0
    if model == 'PD':
        R = b-c
        T = b
        S = -c
        P = 0

    # Fitness values
    f_c = x(G)*(R-S)+S
    f_d = x(G)*(T-P)+P
    f_delta = x(G)*(R-T-S+P)+S-P
    fitness = [f_c, f_d, f_delta]

    return fitness

def setup(N):

    global G
    # Scale-free network
    G = nx.barabasi_albert_graph(N, E, SEED)  

    # Setup with 50% Ds and 50% Cs
    set_node_bool_attribute_with_prob_k(G, 'cooperator', .5) 
    # Game participants each have an initial endowment b
    set_all_node_attributes(G,'endowment', b)
    # Cs contribute a fraction c of their endowment, whereas Ds do not contribute
    set_behavior_node_attributes(G,'contribution', c, 0)

def update(frame):
    # Generate Population of size Z in which individuals engage in an N person dilemma
    N = N_values[frame]

    setup(N)

    # Calculate the degrees of all nodes in the network
    k_s = [k_n for N, k_n in G.degree()]
    k_s = [k_n * random.randint(0, 30) for k_n in k_s]

    # Get unique groups
    groups = set(nx.get_node_attributes(G,'cooperator').values())
    mapping = dict(zip(sorted(groups),count()))
    colors = [mapping[G.nodes[n]['cooperator']] for n in G.nodes()]

    # Visualize
    plt.clf()
    plt.subplot(121)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=k_s, node_color=colors, width=0.5)
    plt.title(f"Network with {N} nodes")

    # Plot the degree distribution as a histogram
    plt.subplot(122)
    plt.hist(k_s, bins=20, edgecolor='k')
    plt.title("Degree Distribution of a Scale-Free Network")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

def evolution_k_with_N():
    # Create a figure for the animation
    fig = plt.figure(figsize=(12, 6))

    # Create the animation
    animation = ani.FuncAnimation(fig, update, frames=len(N_values), repeat=False)

    plt.show()

def evolution_gradient_of_selection_with_x(model):
    setup(N_values[2])

    # Evaluate function and create the plot
    x_vals = [i / 1000 for i in range(1001)] 

    for model in models:
        plt.plot(x_vals, [gradient_of_selection(x,model) for x in x_vals], label = model)

    plt.xlabel('x (Fraction of cooperators)')
    plt.ylabel('Gradient of selection')
    plt.title('Gradient of selection vs. x')
    plt.legend()

    plt.show()

#evolution_k_with_N()
evolution_gradient_of_selection_with_x('H')
