import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("crime1.csv")
column = df["ViolentCrimesPerPop"]

plt.figure()
plt.hist(column, bins=30)

plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")

plt.show()

plt.boxplot(column)

plt.title("Box Plot of Violent Crimes Per Population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Values")

plt.show()
#the histogram shows distribution of violent crimes in different populations.
#The graph seems to be right skewed, indicating that some populations have very high violent crime rates.
#majority of values seem to lie on the lower end, while some populations have very high frequencies of violent crime.
#the boxplot shows the median line to be in the 0.4 value range.
#compared to the maximum value, the median is much lower, showing that the interquartile range is in the low ranges.
#since points outside the maximums and minimums indicate outliers, there do not seem to be any outliers in our graph, however it is worth noting that our maximum value is very high.