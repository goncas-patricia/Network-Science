import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.optimize import newton
from scipy.optimize import fsolve
from scipy.stats import hypergeom
from itertools import count
import numpy as np
import sympy as sp
from sympy.solvers import solve
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


# Notas
# risk loss tem erro
# prestar atenção à diferença entre Z e N
# k has differents meanings on infinite (number of k in N) vs finite (number of k in Z)


# TASKS
# mudar o cost-benefit para N-player
# introdução do risk corretamente (replicator equation paper or slide 43 topic 11 3rd presentation)
# Última parte do paper (Evolutionary Dynamics in Structured Populations)


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
STRUCUTED_POPULATION = 'structured_population'


#####################
### Mischelaneous ###
#####################


random.seed(SEED)


###################
### Global Vars ###
###################


# Nodes, thresholds, group size
N_values = [4, 8, 12, 24, 48] 
M_values = [2, 3, 4, 5]
Z1 = 50 # 1C, 2A-D
Z2 = 500 # 3B-C
# Risk
r = [0.0001, 0.25, 0.50, 0.75, 1.00]
# Models
models = ['H', STAG_HUNT, SNOW_DRIFT, PRISONER_DILEMMA] 
# Endowment, contribution (fraction c of the endowment)
b = 1
c = .1
# Social learning, mutation rate
beta = 0.5
u = 0.01
mutation_matrix = []


############################################################
### Functions: network characteristics and risk of loss ####
############################################################


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


def setup(Z, model):
    global G
    if model == BARABASI_ALBERT:
        # Scale-free network (barabasi-albert)
        G = nx.barabasi_albert_graph(Z, Z//2, SEED)  
    elif model == ERDOS_RENYI:
        # Random Network (erdos-renyi)
        G = nx.erdos_renyi_graph(Z, 1, SEED)
    elif model == COMPLETE:
        # Classic complete connected graph 
        G = nx.classic.complete_graph(Z)

    # Setup with 50% Ds and 50% Cs
    set_node_bool_attribute_with_prob_k(G, COOPERATORS, INITIAL_COOPERATOR_PROB) 
    # Game participants each have an initial endowment b
    set_all_node_attributes(G, ENDOWMENT, b)
    # Cs contribute a fraction c of their endowment, whereas Ds do not contribute
    set_behavior_node_attributes(G, CONTRIBUTION, c, 0)


def Z(G):
    """Returns the number of nodes in the network"""
    return len(G.nodes())


def cooperators(G):
    """Returns a list of cooperators in the population"""
    cooperators = [node for node in G.nodes() if G.nodes[node][COOPERATORS] == 1]
    return cooperators


def number_of_cooperators(G):
    """Number of cooperators in the population"""
    return len(cooperators(G))


def x(G):
    """Fraction of contributors in the population:

    returns number of cooperators / number of nodes"""
    x = number_of_cooperators(G) / len(G.nodes())
    return x


def fraction_of_defectors(G):
    """Fraction of defectors in the population
    returns 1 - fraction_of_contributors(G)"""
    return 1 - x(G)


# def risk_loss(risk, N, M, k_in_N):
#     """Risk loss function:

#     If a group of size N does not contain at least M Cs, 
#     all members will lose their remaining endowments with a probability r"""
#     if k_in_N < M:
#         if random.random() <= risk:
#             for node in G.nodes():
#                 G.nodes[node][ENDOWMENT] = 0


######################################################
### Functions: gradient of selection, fitnesses ######
######################################################


def gradient_of_selection(x, risk, model, N, M, k, pop_type=INFINITE_WELL_MIXED):
    """Gradient of selection:

    Replicator equation
    
    pop_type = type of population (infinite well-mixed or finite well-mixed)"""
    if pop_type == INFINITE_WELL_MIXED:
        return x * (1 - x) * fitness_delta(x, risk, model, N, M, k, pop_type)
    elif pop_type == FINITE_WELL_MIXED:
        return x * (1 - x) * np.tanh(0.5 * beta * fitness_delta(x, risk, model, N, M, k, pop_type))
    

def fitness(x, risk, model, N, M, k, pop_type=INFINITE_WELL_MIXED, Z = 500):
    """Fitness for Cs and Ds
    """

    if model == STAG_HUNT:
        fC = 1
        fD = 0

    elif model == SNOW_DRIFT:
        fC = 1
        fD = 0

    elif model == PRISONER_DILEMMA:
        if pop_type == INFINITE_WELL_MIXED:
            fC, fD = fitness_infinite_well_mixed(x, risk, N, M, k)
        elif pop_type == FINITE_WELL_MIXED:
            fC, fD = fitness_finite_well_mixed(Z, risk, N, M, k)

    return [fC, fD, fC - fD]


def fitness_C(x, risk, model, N, M, k, pop_type=INFINITE_WELL_MIXED, Z=500):
    return fitness(x, risk, model, N, M, k, pop_type, Z)[0]


def fitness_D(x, risk, model, N, M, k, pop_type=INFINITE_WELL_MIXED, Z=500):
    return fitness(x, risk, model, N, M, k, pop_type, Z)[1]
    

def fitness_delta(x, risk, model, N, M, k, pop_type=INFINITE_WELL_MIXED, Z=500):
    """Fitness delta:

    m = threshold of cooperators
    pop_type = type of population (infinite well-mixed or finite well-mixed)

    Fitness of cooperators - fitness of defectors. Based on the formulas provided in the paper:

    Risk of collective failure provides an escape from the tragedy of the commons,
    Francisco C. Santos, Jorge M. Pacheco"""
    if pop_type == INFINITE_WELL_MIXED:
        return fitness_delta_infinite_well_mixed(x, risk, N, M)
    elif pop_type == FINITE_WELL_MIXED:
        return fitness(x, risk, model, N, M, k, pop_type, Z)[2]


def payoffD(k, M, r):
    """Returns the payoff of a single C in a group of K Cs
    M < N is the coordination threshold necessary to achieve a collective benefit"""
    return b * (theta(k - M) + (1 - r) * (1 - theta(k - M)))


def theta(x):
    """Heaviside Step function:
    
    1 if x >= 0, 0 otherwise"""
    return 1 if x >= 0 else 0


#########################################
### Functions: infinite well-mixed ######
#########################################


def fitness_infinite_well_mixed(x, risk, N, M, k):
    """ Computes fitness C, fitness D and fitness Delta for infinite well-mixed populations
        The k stands for the k Cs in the group (of size N)
    """
    fC = 0
    fD = 0

    for k in range(N):
        binomial = math.comb(N - 1, k)
        piD = payoffD(k, M, risk)
        piC = payoffD(k + 1, M, risk) - c*b
        mult = (x ** k) * ((1 - x) ** (N - 1 - k))
        fC += binomial * mult * piC
        fD += binomial * mult * piD

    return fC, fD


def gamma(x, N, M):
    """Auxiliary function for the infinite well-mixed population fitness delta"""
    return math.comb(N - 1, M - 1) * (x**(M - 1)) * ((1 - x)**(N-M))


def fitness_delta_infinite_well_mixed(x, risk, N, M):
    """Computes the fitness delta (fitness contributors - fitness defectors)
    for an infinite well-mixed population, based on the formula provided in the paper:
    
    Risk of collective failure provides an escape from the tragedy of the commons,
    Francisco C. Santos, Jorge M. Pacheco
    
    Used for an infinite number of nodes.
    """
    return b * (gamma(x, N, M)*risk - c)


#######################################
### Functions: finite well-mixed ######
#######################################


def fitness_finite_well_mixed(Z, risk, N, M, k):
    """ Computes fitness C, fitness D and fitness Delta for finite well-mixed populations of size Z
        The k stands for the k Cs in the population (of size Z)
    """

    fC = 0
    fD = 0

    binomialMain = math.comb(Z - 1, N - 1)
    for j in range(min(N, k)):
        piD = payoffD(j, M, risk)
        piC = payoffD(j + 1, M, risk) - c*b
        # print("")
        # print("Z =", Z)
        # print("k =", k)
        # print("N =", N)
        # print("j =", j)
        # print("Z - k =", Z - k)
        # print("N - j - 1 =", N - j - 1)
        binomialC = math.comb(k - 1, j) * math.comb(Z - k, N - j - 1)
        binomialD = math.comb(k, j) * math.comb(Z - k - 1, N - j - 1)
        fC += binomialC * piD
        fD += binomialD * piC
    fC /= binomialMain
    fD /= binomialMain
    return fC, fD


####################################################
### Functions: stochastic effects (imitation) ######
####################################################

def stochastic_birth_death(G, risk, model, N, M, mutation_matrix, pop_type=FINITE_WELL_MIXED, num_iterations=1):
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
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    state = 0 # positive if increase in contributors > increase in defectors
              # negative if it is smaller

    k = number_of_cooperators(G)
    for i in range(num_iterations):
        # Select node that will adopt its strategy
        social_node = random.choice(nodes)

        # Select a random node
        random_node = random.choice(nodes)

        x = k / num_nodes
        fitnessContributor, fitnessDefector = fitness(x, risk, model, N, M, pop_type)
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
        adoption_prob = 1 / (1 + np.exp(-beta * (fitness_random_node - fitness_current_node)))

        # The current node adopts the strategy of the random node with probability prob
        if random.random() <= adoption_prob:
            if G.nodes[social_node][COOPERATORS] != G.nodes[random_node][COOPERATORS]:
                # If they had different strategies, update the state
                if G.nodes[social_node][COOPERATORS] == 1:
                    state += 1
                    k += 1
                else:
                    state -= 1
                    k -= 1

        # Node mutates its strategy with mutation rate u
        # according to the mutation matrix
        contribute_prob = mutation_matrix[k][k + 1]
        defect_prob = mutation_matrix[k][k - 1]
        mutation_prob = random.random()
        if G.nodes[social_node][COOPERATORS] == 1:
            if mutation_prob <= defect_prob:
                G.nodes[social_node][COOPERATORS] = 0
                state -= 1
                k -= 1
        else:
            if mutation_prob <= contribute_prob:
                G.nodes[social_node][COOPERATORS] = 1
                state += 1
                k += 1

    return state


def iterative_stochastic_birth_death(G, Z, x, risk, model, N, M, pop_type=FINITE_WELL_MIXED,
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

    mutation_matrix = tridiagonal_matrix_algorithm(Z, N, M, risk, model)
    
    state = 0
    converged = False
    timesCloseToConvergence = 0
    iteration = 0
    while not converged and iteration < max_iterations:
        # TODO - Not sure how the number of players might be made to affect this...

        variation = stochastic_birth_death(G, risk, model, N, M, pop_type=pop_type, num_iterations=100)
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


##########################################
### Functions: behavioral mutations ######
##########################################


def tridiagonal_matrix_algorithm(Z, N, M, risk, model):
    transition_matrix = np.zeros((Z, Z))

    for k in range(Z):
        x = k/Z
        pk_k_plus_1 = prob_contributor_increase_mutation(x, Z, risk, model, N, M, k)
        pk_k_minus_1 = prob_contributor_decrease_mutation(x, Z, risk, model, N, M, k)
        pk_k = 1 - pk_k_minus_1 - pk_k_plus_1
        for j in range(Z):
            if k == j:
                transition_matrix[k][j] = pk_k
            elif (k + 1) == j:
                transition_matrix[k][j] = pk_k_plus_1
            elif (k - 1) == j:
                transition_matrix[k][j] = pk_k_minus_1
            else:
                transition_matrix[k][j] = 0

    transition_matrix = transition_matrix.T

    return transition_matrix


def prob_contributor_increase_mutation(x, Z, risk, model, N, M, k, pop_type=FINITE_WELL_MIXED):
    """Probability of contributor increase due to mutation:

    Returns the probability of an increase in contributors due to mutation"""

    return ( (1 - u) 
            * prob_increase_and_decrease_number_Cs_by_one(x, Z, risk, model, N, M, k, pop_type=pop_type, increase=True)
            + u * (Z - k) / Z
    )


def prob_contributor_decrease_mutation(x, Z, risk, model, N, M, k, pop_type=FINITE_WELL_MIXED):
    """Probability of contributor decrease due to mutation:

    Returns the probability of a decrease in contributors due to mutation"""

    return ( (1 - u) 
            * prob_increase_and_decrease_number_Cs_by_one(x, Z, risk, model, N, M, k, pop_type=pop_type, increase=False)
            + u * k / Z
    )


def prob_increase_and_decrease_number_Cs_by_one(x, Z, risk, model, N, M, k, pop_type=FINITE_WELL_MIXED, increase=True):
    """Probability of increase and decrease in the number of contributors by one:

    Returns the probability of an increase and decrease in the number of contributors by one"""
    k_over_Z = k / Z
    aux_term = (Z - k) / Z
    sign = -1
    if not increase:
        sign = 1

    fitness_C , fitness_D, fitness_delta = fitness(x, risk, model, N, M, k, pop_type, Z)
    return k_over_Z * aux_term * (1 + math.exp(sign * (fitness_C - fitness_D)))**(-1)


def stationary_distribution(Z, N, M, risk, model):
    '''
    Compute the stationary distribution P(k/Z)
    of the complete Markov chain with Z + 1 states
    (as shown in Figs. 1C, 2A and 2B).
    '''

    S = tridiagonal_matrix_algorithm(Z, N, M, risk, model)
    # Note: discuss if .T should be next to S
    eigenvalues, eigenvectors = np.linalg.eig(S)

    last_diff = float("inf")
    for idx, eigval in enumerate(eigenvalues):
        if abs(eigval - 1) < last_diff:
            close_to_1_idx = idx
            last_diff = abs(eigval - 1)

    target_eigenvector = eigenvectors[:, close_to_1_idx].real
    stationary_distribution = target_eigenvector / sum(target_eigenvector) 
    print("S: ", S)
    print("target eigenvector:", target_eigenvector)
    print("target eigenvalue:", eigenvalues[close_to_1_idx])
    return stationary_distribution


############################################
### Functions: related to the figures ######
############################################


def cost_to_risk_ratio(risk):
    """Cost to risk ratio:

    Cost to risk ratio of the game
    
    i = index of the risk value in the risk array r = [0.00, 0.25, 0.50, 0.75, 1.00]"""
    if risk == 0:
        raise TypeError("cost_to_risk_ratio: risk must be different than 0")

    return c / risk


def internal_roots(risk, N, M):
    x = sp.symbols('x', real = True)
    equation = cost_to_risk_ratio(risk) - gamma(x, N, M)
    solution = solve(equation) #, x0 = .5)
    return solution


#############################
### Functions: figures ######
#############################


def evolution_k_with_N():
    """ Create an animation in which N grows
    """
    # Create a figure for the animation
    fig = plt.figure(figsize=(12, 6))

    # Create the animation
    animation = ani.FuncAnimation(fig, update, frames=len(N_values), repeat=False)

    plt.show()


def evolution_gradient_of_selection_with_x(model):
    """ Gives us figure 1A/1.A.
        Infinite well-mixed population.
    """

    # Set the constants and variable x
    N = 6
    M = 3
    x_vals = [i / 1000 for i in range(1001)] 

    for risk in r:
        plt.plot(x_vals, [gradient_of_selection(x, risk, model, N, M, int(N*x), INFINITE_WELL_MIXED) for x in x_vals], label = risk)
    plt.plot(x_vals, [0 for x in x_vals])

    plt.legend()
    plt.xlabel('x (Fraction of cooperators)')
    plt.ylabel('Gradient of selection')
    plt.title('Gradient of selection vs. x')

    plt.show()

def evolution_gamma_with_gradient_of_selection():
    """ Gives us figure 1B/1.B.
        Infinite well-mixed population.
        It also retrieves the internal roots of the gradient of selection.
    """
    N = 6

    # Evaluate function and create the plot
    x_vals = [i / 1000 for i in range(1001)]

    for M in M_values:
        plt.plot(x_vals, [gamma(x, N, M) for x in x_vals], label = f'M = {M}')

    for risk in r[1:]:
        plt.plot(x_vals, [cost_to_risk_ratio(risk) for x in x_vals], label=f'Risk = {risk}')   

    plt.legend()
    plt.xlabel('Gradient of selection')
    plt.ylabel('Gamma')
    plt.title(f'Gamma vs gradient of selection (N = {N})')

    plt.show()

    internal_roots_dic = dict()
    # Make sure we can solve this (if needed)
    for risk in r:
        for M in M_values:
            key = (risk, M)
            try:
                root = internal_roots(risk, N, M)
                internal_roots_dic[key] = root
                print(f"Roots for N = {N}, M = {M}, risk = {risk}: ", root,"\n")
            except Exception as e: 
                raise e
    print("internal roots: ", internal_roots_dic)

def evolution_stationary_distribution_with_x(model='PD'):
    """ Gives us figure 1C/1.C.
        For finite populations.
    """

    Z = 50
    N = 6
    M = 3
    x_vals =  [i/Z for i in range(Z)]

    # Reduzimos a matrix tridiagonal para ZxZ

    for risk in r:
        plt.plot(x_vals, [P for P in stationary_distribution(Z, N, M, risk, model)], label = risk)
    
    plt.legend()
    plt.xlabel('x (Fraction of cooperators)')
    plt.ylabel('Satationary distribution')
    plt.title('Stationary distribution vs. x')

    plt.show()


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



#evolution_k_with_N()
#evolution_gradient_of_selection_with_x('PD')
evolution_gamma_with_gradient_of_selection()
#evolution_stationary_distribution_with_x()