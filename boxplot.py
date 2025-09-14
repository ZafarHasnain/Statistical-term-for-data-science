'''A box plot (also called a box-and-whisker plot) is a graphical way to show the distribution of a dataset. It displays the five-number summary of the data:

Minimum – the smallest value (excluding outliers)

Q1 (First Quartile) – 25% of the data is below this value

Median (Q2) – the middle value (50% of the data is below)

Q3 (Third Quartile) – 75% of the data is below this value

Maximum – the largest value (excluding outliers)'''

import matplotlib.pyplot as plt

data = [7, 15, 36, 39, 42, 43, 46, 49, 100]
plt.boxplot(data)
plt.title("Box Plot Example")
plt.show()
