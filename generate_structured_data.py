import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Number of hotels
num_hotels = 100

hotel_names = [f"Hotel {i}" for i in range(1, num_hotels + 1)]
hotel_names = np.random.choice(hotel_names, num_hotels)

star_ratings = np.random.randint(1, 6, num_hotels)

distances = np.random.randint(100, 5001, num_hotels)

# Normalize distances to a range of 0-1 to reduce the impact of outliers
distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())

price_noise = np.random.randint(-20, 21, num_hotels)  # Adjust the range as needed

# Calculate prices based on a combination of star ratings and normalized distances
# You can adjust the coefficients as needed
prices = 50 + star_ratings * 50 - distances_normalized * 200 + price_noise

# Adjust prices to fall within the range of 50-500
min_price = 50
max_price = 500

# Calculate the min and max values of the calculated prices
min_calculated_price = min(prices)
max_calculated_price = max(prices)

# Scale prices to fit the desired range
prices = ((prices - min_calculated_price) / (max_calculated_price - min_calculated_price)) * (max_price - min_price) + min_price
prices = np.round(prices).astype(int)

# Create a DataFrame
data = {
    "Hotel Name": hotel_names,
    "Hotel Star Rating": star_ratings,
    "Distance to City Center": distances,
    "Price": prices
}

df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('fake_structured_data.csv', index=False)