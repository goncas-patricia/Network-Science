import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import random

SEED = 7
N_values = list(range(10, 100))  # Total number of nodes
E = 5 # Control the scale-free properties

def update(frame):

    # Generate
    N = N_values[frame]
    G = nx.barabasi_albert_graph(N, E, SEED)  # Scale-free network

    # Visualize
    # Plot the network
    plt.clf()
    plt.subplot(121)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=50, node_color=G.nodes(), width=0.5)
    plt.title(f"Network with {N} nodes")

    # Calculate the degrees of all nodes in the network
    k_s = [k_n for N, k_n in G.degree()]
    k_s = [k_n * random.randint(0, 1) for k_n in k_s]

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

