import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt

# Example dataset
data = pd.DataFrame({
    'Bread': [1,0,1,1,0],
    'Butter': [1,1,0,1,0],
    'Milk': [0,1,1,1,1]
})

# Frequent itemsets
itemsets = apriori(data, min_support=0.4, use_colnames=True)

# Rules
rules = association_rules(itemsets, metric="lift", min_threshold=1)

# Convert frozensets to strings for plotting
rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])

# Create graph
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents', 
                            edge_attr=True, create_using=nx.DiGraph())

# Draw graph
plt.figure(figsize=(6,4))
pos = nx.spring_layout(G, seed=42)  # layout algorithm
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, 
        font_size=10, font_weight="bold", arrows=True)
nx.draw_networkx_edge_labels(G, pos, 
                             edge_labels={(row['antecedents'], row['consequents']): 
                                          f"{row['confidence']:.2f}" 
                                          for idx, row in rules.iterrows()},
                             font_size=8)
plt.show()
