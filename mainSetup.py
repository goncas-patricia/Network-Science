import networkx as nx
import matplotlib.pyplot as plt
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


# Generate
E = 8  # Adjust this value to control the scale-free properties
N = 30  # Total number of nodes
G = nx.barabasi_albert_graph(N, E, SEED)

# Visualize
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=30, width=0.5)
plt.show()

# Calculate the degrees of all nodes in the network
k_s = [k_n for N, k_n in G.degree()]

# Plot the degree distribution as a histogram
plt.hist(k_s, bins=20, edgecolor='k')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()



set_node_bool_attribute_with_prob_k(G, 'defector', INITIAL_DEFECTOR_PROB)

print(G.nodes(data=True))