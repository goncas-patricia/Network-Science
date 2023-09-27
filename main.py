import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from itertools import count
import numpy as np
import random
import math

# TASKS
# mudar o cost-benefit para N-player
# introdução do risk corretamente (replicator equation paper or slide 43 topic 11 3rd presentation)

#################
### Constants ###
#################

SEED = 7
INITIAL_COOPERATOR_PROB = 0.5
PRISONER_DILEMMA = 'PD'
STAG_HUNT = 'SH'
SNOW_DRIFT = 'SG'
CHICKEN_GAME = SNOW_DRIFT
BARABASI_ALBERT = 'BA'
ERDOS_RENYI = 'ER'
COMPLETE = 'COMPLETE'
COOPERATORS = 'cooperator'
ENDOWMENT = 'endowment'
CONTRIBUTION = 'contribution'


#####################
### Mischelaneous ###
#####################

random.seed(SEED)


###################
### Global Vars ###
###################

# Total number of nodes
N_values = list(range(2, 100)) 
# Risk
r = [0.00, 0.25, 0.50, 0.75, 1.00]
# Models
models = ['H',STAG_HUNT, SNOW_DRIFT, PRISONER_DILEMMA] 
# Initial endowment
b = 1
# Contribution (fraction of the endowment)
c = .1*b
# Social learning
beta = 0.5
# Threshold (fraction of N(G))
m = 0.5

##################
### Functions ####
##################

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
        if G.nodes[node][COOPERATORS] == 1:
            G.nodes[node][attr_name] = cooperator
        else:
            G.nodes[node][attr_name] = defector


def theta(x):
    """Heaviside Step function
    
    1 if x >= 0, 0 otherwise"""
    return 1 if x >= 0 else 0


# Group of size N and k Cs
def payoffC(k, M, r):
    return b * (theta(k - M) + (1 - r) * (1 - theta(k - M)))


def N(G):
    return len(G.nodes())


def cooperators(G):
    cooperators = [node for node in G.nodes() if G.nodes[node][COOPERATORS] == 1]
    return cooperators


def number_of_cooperators(G):
    return len(cooperators(G))


def fraction_of_contributors(G):
    """Fraction of contributors in the population
    returns number of cooperators / number of nodes"""
    x = number_of_cooperators(G) / len(G.nodes())
    return x


def fraction_of_defectors(G):
    """Fraction of defectors in the population
    returns 1 - fraction_of_contributors(G)"""
    return 1 - fraction_of_contributors(G)


def risk_loss(G, M):
    """Risk loss function
    If the number of cooperators is less than the threshold M,
    then all nodes lose their endowment with probability r[2]"""
    if number_of_cooperators(G) < M:
        if random.random() <= r[2]:
            for node in G.nodes():
                G.nodes[node][ENDOWMENT] = 0


def gradient_of_selection(x, model):
    # 2-Person
    #return x * (1 - x) * fitness(x,model)[2]
    # Finite well-mixed populations
    return x * (1 - x) * np.tanh(0.5 * beta * fitness(x,model)[2])


def fitness(x, model):
    # Cost-Benefit Values 
    # Para já, estáticos e 2-Player (should be N-Player)
    if model == 'H':
        R = b
        T = b-c
        S = b-c
        P = 0
    elif model == STAG_HUNT:
        R = b
        T = c
        S = -c
        P = 0
    elif model == SNOW_DRIFT:
        R = 1
        T = b+c
        S = c
        P = 0
    elif model == PRISONER_DILEMMA:
        R = b-c
        T = b
        S = -c
        P = 0

    # Fitness
    fC = 0
    fD = 0

    numVertices = N(G)
    for k in range(numVertices):
        binomial = math.comb(numVertices - 1, k)
        payoffC = payoffC(x*numVertices + 1, m*numVertices, r[2])
        mult = (x ** k) * ((1 - x) ** (numVertices - 1 - k))
        fC += binomial * mult * payoffC
        fD = binomial * mult * (payoffC - c*b)

    fDelta = x*(R-T-S+P)+S-P
    fitness = [fC, fD, fDelta]

    return fitness


def setup(N, model):
    global G
    if model == BARABASI_ALBERT:
        # Scale-free network (barabasi-albert)
        G = nx.barabasi_albert_graph(N, N//2, SEED)  
    elif model == ERDOS_RENYI:
        # Random Network (erdos-renyi)
        G = nx.erdos_renyi_graph(N, 1, SEED)
    elif model == COMPLETE:
        # Complete/fully connected graph 
        G = nx.classic.complete_graph(N)

    # Setup with 50% Ds and 50% Cs
    set_node_bool_attribute_with_prob_k(G, COOPERATORS, INITIAL_COOPERATOR_PROB) 
    # Game participants each have an initial endowment b
    set_all_node_attributes(G, ENDOWMENT, b)
    # Cs contribute a fraction c of their endowment, whereas Ds do not contribute
    set_behavior_node_attributes(G, CONTRIBUTION, c, 0)


def update(frame):
    # Generate Population of size Z in which individuals engage in an N person dilemma
    N = N_values[frame]

    setup(N, BARABASI_ALBERT)

    # Calculate the degrees of all nodes in the network
    k_s = [k_n for N, k_n in G.degree()]
    k_s = [k_n * random.randint(0, 30) for k_n in k_s]

    # Get unique groups
    groups = set(nx.get_node_attributes(G,COOPERATORS).values())
    mapping = dict(zip(sorted(groups),count()))
    colors = [mapping[G.nodes[n][COOPERATORS]] for n in G.nodes()]

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
    setup(N_values[2], ERDOS_RENYI)

    # Evaluate function and create the plot
    x_vals = [i / 1000 for i in range(1001)] 

    for model in models:
        plt.plot(x_vals, [gradient_of_selection(x,model) for x in x_vals], label = model)

    plt.xlabel('x (Fraction of cooperators)')
    plt.ylabel('Gradient of selection')
    plt.title('Gradient of selection vs. x')
    plt.legend()

    plt.show()

evolution_k_with_N()
evolution_gradient_of_selection_with_x('H')
