import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the structured hotel data
df = pd.read_csv('fake_structured_data.csv')

# Create a scatter plot of Distance vs. Price
plt.figure(figsize=(10, 5))
plt.scatter(df['Distance to City Center'], df['Price'], c='b', marker='o', alpha=0.5)
plt.title('Distance to Price')
plt.xlabel('Distance to City Center')
plt.ylabel('Price')
plt.grid(True)

# Set the x-axis to display whole numbers with a step of 500
plt.xticks(np.arange(0, 5001, step=500))

# Show the plot
plt.show()

# Create a scatter plot of Star Rating vs. Price with whole numbers for star ratings
plt.figure(figsize=(10, 5))
plt.scatter(df['Hotel Star Rating'], df['Price'], c='r', marker='s', alpha=0.5)
plt.title('Star Rating to Price')
plt.xlabel('Hotel Star Rating')
plt.ylabel('Price')
plt.grid(True)

# Adjust the x-axis to display only whole numbers for star ratings
plt.xticks(np.arange(1, 6, step=1))

# Show the plot
plt.show()