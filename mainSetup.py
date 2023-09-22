import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from itertools import count
import random

SEED = 7
INITIAL_DEFECTOR_PROB = 0.6
random.seed(SEED)

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

N_values = list(range(10, 100))  # Total number of nodes
E = 5 # Control the scale-free properties

def update(frame):

    # Generate
    N = N_values[frame]
    G = nx.barabasi_albert_graph(N, E, SEED)  # Scale-free network

    set_node_bool_attribute_with_prob_k(G, 'defector', .5)

    # Calculate the degrees of all nodes in the network
    k_s = [k_n for N, k_n in G.degree()]
    k_s = [k_n * random.randint(0, 30) for k_n in k_s]

    # get unique groups
    groups = set(nx.get_node_attributes(G,'defector').values())
    mapping = dict(zip(sorted(groups),count()))
    colors = [mapping[G.nodes[n]['defector']] for n in G.nodes()]

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

    # Animation
    #animator = ani.FuncAnimation()

    # Ask the user if they want to continue the simulation
    #if input("Continue? (yes/no): ").lower() != "yes":
    #    flag = 0

    plt.pause(0.01)

# Create a figure for the animation
fig = plt.figure(figsize=(12, 6))

# Create the animation
animation = ani.FuncAnimation(fig, update, frames=len(N_values), repeat=False)

plt.show()