import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Example dataset
dataset = [
    ['milk', 'bread', 'butter'],
    ['milk', 'bread'],
    ['milk', 'apple'],
    ['bread', 'butter'],
    ['milk', 'bread', 'apple', 'butter']
]

# Convert to one-hot encoding
df = pd.DataFrame([{item: (item in transaction) 
                    for item in set(sum(dataset, []))} 
                   for transaction in dataset])

# Frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Association rules with all metrics
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']]

print(rules)
