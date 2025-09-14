import networkx as nx
import matplotlib.pyplot as plt

# Create a social network graph
G = nx.Graph()

# Add people (nodes)
G.add_nodes_from(["Alice", "Bob", "Charlie", "David", "Eva", "Frank"])

# Add relationships (edges)
G.add_edges_from([
    ("Alice", "Bob"),
    ("Alice", "Charlie"),
    ("Bob", "David"),
    ("Charlie", "David"),
    ("David", "Eva"),
    ("Eva", "Frank"),
    ("Charlie", "Frank")
])

# Calculate centrality
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)

# Draw the network
plt.figure(figsize=(6,5))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): "Friend" for u,v in G.edges()}, font_size=8)
plt.title("Social Network Graph")
plt.show()
