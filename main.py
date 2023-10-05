import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.stats import hypergeom
from itertools import count
import numpy as np
import random
import math

# Coisas que podem ser úteis/interessantes falar na apresentação/workshop:
# - O que é a tragédia dos comuns
#       - O que é o risco de falha coletiva
# - Exemplos reais da tragédia dos comuns
# - Paper (parte dos métodos Evolutionary Dynamics in Finite Well-Mixed Populations)
#   fala de random drift para low fitness values. Pode ser interessante explicar o que
#   isso quer dizer e relacionar com o conceito análogo na biologia (aka driva genética)
#   na fixação de um fenótipo/alelo
# - Observações dos resultados empíricos


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
INFINITE_WELL_MIXED = 'infinite_well_mixed'
FINITE_WELL_MIXED = 'finite_well_mixed'


#####################
### Mischelaneous ###
#####################

random.seed(SEED)


###################
### Global Vars ###
###################

# Total number of nodes
N_values = list(range(2, 100)) 
Z = 150
# Risk
r = [0.00, 0.25, 0.50, 0.75, 1.00]
# Models
models = ['H', STAG_HUNT, SNOW_DRIFT, PRISONER_DILEMMA] 
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
    """Heaviside Step function:
    
    1 if x >= 0, 0 otherwise"""
    return 1 if x >= 0 else 0


# Group of size N and k Cs
def payoffD(k, M, r):
    """Returns the payoff of a single C in a group of K Cs
    M < N is the coordination threshold necessary to achieve a collective benefit"""
    return b * (theta(k - M) + (1 - r) * (1 - theta(k - M)))


def N(G):
    """Returns the number of nodes in the network"""
    return len(G.nodes())


def cooperators(G):
    """Returns a list of cooperators in the population"""
    cooperators = [node for node in G.nodes() if G.nodes[node][COOPERATORS] == 1]
    return cooperators


def number_of_cooperators(G):
    """Number of cooperators in the population"""
    return len(cooperators(G))


def fraction_of_contributors(G):
    """Fraction of contributors in the population:

    returns number of cooperators / number of nodes"""
    x = number_of_cooperators(G) / len(G.nodes())
    return x


def fraction_of_defectors(G):
    """Fraction of defectors in the population
    returns 1 - fraction_of_contributors(G)"""
    return 1 - fraction_of_contributors(G)


def risk_loss(G, risk, M):
    """Risk loss function:

    If the number of cooperators is less than the threshold M,
    then all nodes lose their endowment with probability risk"""
    if number_of_cooperators(G) < M:
        if random.random() <= risk:
            for node in G.nodes():
                G.nodes[node][ENDOWMENT] = 0


def gradient_of_selection(x, risk, model, pop_type=INFINITE_WELL_MIXED):
    """Gradient of selection:

    Replicator equation
    
    pop_type = type of population (infinite well-mixed or finite well-mixed)"""
    # 2-Person
    #return x * (1 - x) * fitness(x,model)[2]
    if pop_type == INFINITE_WELL_MIXED:
        return x * (1 - x) * fitness_delta(x, risk, model, m, pop_type)
    elif pop_type == FINITE_WELL_MIXED:
        return x * (1 - x) * np.tanh(0.5 * beta * fitness(x, risk, model)[2])


def fitness(x, risk, model, pop_type=INFINITE_WELL_MIXED):
    """Fitness for Cs and Ds

    First: 2-Player for each model
    
    Second: N-Player for the model present in the paper"""

    # Fitness
    fC = 0
    fD = 0

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
        numVertices = N(G)
        if pop_type == INFINITE_WELL_MIXED:
            for k in range(numVertices):
                binomial = math.comb(numVertices - 1, k)
                payoffD = payoffD(x*numVertices, m*numVertices, risk)
                payoffC = payoffD(x*numVertices + 1, m*numVertices, risk) - c*b
                mult = (x ** k) * ((1 - x) ** (numVertices - 1 - k))
                fC += binomial * mult * payoffD
                fD += binomial * mult * payoffC
        elif pop_type == FINITE_WELL_MIXED:
            for k in range(numVertices):
                j = k + 1
                while j > k:
                    j = int(random.random()*numVertices)
                binomialC = math.comb(k, j) * math.comb(Z - k - 1, numVertices - j - 1)
                binomialD = math.comb(k - 1, j) * math.comb(Z - k, numVertices - j - 1)
                payoffD = payoffD(x*numVertices, m*numVertices, risk)
                payoffC = payoffD(x*numVertices + 1, m*numVertices, risk) - c*b
                fC += binomialC * payoffD
                fD += binomialD * payoffC

            binomialMain = math.comb(Z - 1, numVertices - 1)
            fC /= binomialMain
            fD /= binomialMain

    fitness = [fC, fD, fC - fD]

    return fitness


def cost_to_risk_ratio(i):
    """Cost to risk ratio:

    Cost to risk ratio of the game
    
    i = index of the risk value in the risk array r = [0.00, 0.25, 0.50, 0.75, 1.00]"""
    if type(i) != int:
        raise TypeError("cost_to_risk_ratio: i must be an integer")
    if i < 0 or i > len(r):
        raise ValueError("cost_to_risk_ratio: i must be between 0 and len(r)")

    return c / r[i]


def _aux_infinite_well_mixed(x, m):
    """Auxiliary function for the infinite well-mixed population fitness delta"""
    numVertices = N(G)
    M = int(m*numVertices)
    return math.comb(numVertices - 1, M - 1) * (x**(M - 1)) * ((1 - x)**(numVertices-M))


def fitness_delta_infinite_well_mixed(x, risk, m):
    """Computes the fitness delta (fitness contributors - fitness defectors)
    for an infinite well-mixed population, based on the formula provided in the paper:
    
    Risk of collective failure provides an escape from the tragedy of the commons,
    Francisco C. Santos, Jorge M. Pacheco"""
    return b * (_aux_infinite_well_mixed(x, m)*risk - c)


def fitness_delta(x, risk, model, m, pop_type=INFINITE_WELL_MIXED):
    """Fitness delta:

    m = threshold of cooperators
    pop_type = type of population (infinite well-mixed or finite well-mixed)

    Fitness of cooperators - fitness of defectors. Based on the formulas provided in the paper:

    Risk of collective failure provides an escape from the tragedy of the commons,
    Francisco C. Santos, Jorge M. Pacheco"""
    if pop_type == INFINITE_WELL_MIXED:
        return fitness_delta_infinite_well_mixed(x, risk, m)
    elif pop_type == FINITE_WELL_MIXED:
        pop_size = N(G) # pop_size is called Z in the paper
        pass # TODO implement finite well-mixed population


def stochastic_birth_death_over_all_nodes(G, risk, model, pop_type=FINITE_WELL_MIXED):
    """Stochastic birth-death process over all nodes:
    
    Under pair-wise comparison, each individual i adopts the
    strategy (cooperation or defection) of a randomly selected member of
    the population j, with probability given by the Fermi function. Makes one
    pass over all nodes in the network, updating the state of the population
    
    returns the state of the population after the stochastic birth-death process

    state = positive if increase in contributors > increase in defectors,
            0 if increase in contributors = increase in defectors,
            negative otherwise
    """
    nodes = G.nodes()
    state = 0 # positive if increase in contributors > increase in defectors
              # negative if it is smaller
    for node in nodes:
        # Select a random node
        random_node = random.choice(list(nodes))

        x = fraction_of_contributors(G)
        fitnessContributor, fitnessDefector, delta = fitness(x, risk, model, pop_type=pop_type)
        # TODO - chamar sempre fitness() aqui parece computacionalmente caro
        # Será que abdicamos de alguma precisão e calculamos o fitness fora do loop?

        # Get the fitness of the random node
        if G.nodes[random_node][COOPERATORS] == 1:
            fitness_random_node = fitnessContributor
        else:
            fitness_random_node = fitnessDefector
        # Get the fitness of the current node
        if G.nodes[node][COOPERATORS] == 1:
            fitness_current_node = fitnessContributor
        else:
            fitness_current_node = fitnessDefector

        # Get the probability of the current node adopting the strategy of the random node
        # based on the Fermi function
        prob = 1 / (1 + np.exp(-beta * (fitness_random_node - fitness_current_node)))

        # The current node adopts the strategy of the random node with probability prob
        if random.random() <= prob:
            if G.nodes[node][COOPERATORS] != G.nodes[random_node][COOPERATORS]:
                # If they had different strategies, update the state
                G.nodes[node][COOPERATORS] = G.nodes[random_node][COOPERATORS]
                if G.nodes[node][COOPERATORS] == 1:
                    state += 1
                else:
                    state -= 1

    return state


def stochastic_birth_death(G, risk, model, pop_type=FINITE_WELL_MIXED, num_iterations=1):
    """Stochastic birth-death process:
    
    Under pair-wise comparison, each individual i adopts the
    strategy (cooperation or defection) of a randomly selected member of
    the population j, with probability given by the Fermi function. Makes one
    pass over all nodes in the network, updating the state of the population. Performs
    this process num_iterations times
    
    returns the state of the population after the stochastic birth-death process

    state = positive if increase in contributors > increase in defectors,
            0 if increase in contributors = increase in defectors,
            negative otherwise
    """
    nodes = G.nodes()
    state = 0 # positive if increase in contributors > increase in defectors
              # negative if it is smaller

    for i in range(num_iterations):
        # Select node that will adapt its strategy
        social_node = random.choice(list(nodes))

        # Select a random node
        random_node = random.choice(list(nodes))

        x = fraction_of_contributors(G)
        fitnessContributor, fitnessDefector, delta = fitness(x, risk, model, pop_type=pop_type)
        # TODO - chamar sempre fitness() aqui parece computacionalmente caro
        # Será que abdicamos de alguma precisão e calculamos o fitness fora do loop?

        # Get the fitness of the random node
        if G.nodes[random_node][COOPERATORS] == 1:
            fitness_random_node = fitnessContributor
        else:
            fitness_random_node = fitnessDefector
        # Get the fitness of the current node
        if G.nodes[social_node][COOPERATORS] == 1:
            fitness_current_node = fitnessContributor
        else:
            fitness_current_node = fitnessDefector

        # Get the probability of the current node adopting the strategy of the random node
        # based on the Fermi function
        prob = 1 / (1 + np.exp(-beta * (fitness_random_node - fitness_current_node)))

        # The current node adopts the strategy of the random node with probability prob
        if random.random() <= prob:
            if G.nodes[social_node][COOPERATORS] != G.nodes[random_node][COOPERATORS]:
                # If they had different strategies, update the state
                if G.nodes[social_node][COOPERATORS] == 1:
                    state += 1
                else:
                    state -= 1

    return state


def iterative_stochastic_birth_death(G, risk, model, pop_type=FINITE_WELL_MIXED,
                                     convergence_threshold=10, epsilon=0.0001, 
                                     max_iterations=math.inf):
    """Iterative stochastic birth-death process:
    
    Performs a stochastic birth-death process until the state of the population
    is 0 (no change in the number of contributors or defectors) or seems to
    converge. A value for max_iterations can be provided if convergence is not
    essential.
    
    We say it converges if the variation in the number of contributors or defectors
    is smaller than epsilon for convergence_threshold iterations
    
    Returns state = overall variation in the number of contributors or defectors"""
    state = 0
    converged = False
    timesCloseToConvergence = 0
    iteration = 0
    while not converged and iteration < max_iterations:
        variation = stochastic_birth_death(G, risk, model, pop_type=pop_type, num_iterations=100)
        state += variation

        if abs(variation) < epsilon:
            timesCloseToConvergence += 1
        elif timesCloseToConvergence > 0:
            # We want a streak of timesCloseToConvergence iterations with variation < epsilon
            timesCloseToConvergence -= 1

            # TODO - we could consider only subtracting with a certain probability,
            # but I think this is fine

        if timesCloseToConvergence == convergence_threshold:
            converged = True

        iteration += 1

    return state



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
    setup(N_values[10], ERDOS_RENYI)

    # Evaluate function and create the plot
    x_vals = [i / 1000 for i in range(1001)] 

    for risk in r:
        plt.plot(x_vals, [gradient_of_selection(x, risk, model) for x in x_vals], label = risk)
    
    plt.legend()
    plt.xlabel('x (Fraction of cooperators)')
    plt.ylabel('Gradient of selection')
    plt.title('Gradient of selection vs. x')

    plt.show()

#evolution_k_with_N()
evolution_gradient_of_selection_with_x('PD')
