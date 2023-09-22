import networkx as nx
import matplotlib.pyplot as plt

SEED = 7

# Generate
E = 20  # Adjust this value to control the scale-free properties
N = 200  # Total number of nodes
scale_free_network = nx.barabasi_albert_graph(N, E, SEED)

# Visualize
pos = nx.spring_layout(scale_free_network)
nx.draw(scale_free_network, pos, with_labels=False, node_size=30)
plt.show()

# Calculate the degrees of all nodes in the network
k_s = [k_n for N, k_n in scale_free_network.degree()]

# Plot the degree distribution as a histogram
plt.hist(k_s, bins=20, edgecolor='k')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()
