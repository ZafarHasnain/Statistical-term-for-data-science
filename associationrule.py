from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Example basket data
data = pd.DataFrame({
    'Bread': [1,0,1,1,0],
    'Butter': [1,1,0,1,0],
    'Milk': [0,1,1,1,1]
})

# Find frequent itemsets
frequent_itemsets = apriori(data, min_support=0.4, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Scatter plot
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()

# Graph visualization
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')
nx.draw(G, with_labels=True)
plt.show()
