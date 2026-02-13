import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("crime1.csv")
crimes_col = df["ViolentCrimesPerPop"]

mean_value = crimes_col.mean()
median_value = crimes_col.median()
std_value = crimes_col.std()
min_value = crimes_col.min()
max_value = crimes_col.max()

print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_value)
print("Minimum:", min_value)
print("Maximum:", max_value)

plt.hist(crimes_col, bins=30)

plt.title("Distribution of ViolentCrimesPerPop")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")

plt.show()
#from the printed graph, the distribution appears to be skewed to the right. This is likely because there are extreme values of the mean skewing the graph
#the mean is always more affected by extreme values, as it considers the average of all data points, while median is more resistant to skewness because it depends on middle values.
