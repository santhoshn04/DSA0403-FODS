from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

# Sample dataset of used cars with their respective prices
# Replace this with your own dataset
# Format: [mileage, age, brand, engine type, price]
dataset = [
    [20000, 2, 1, 0, 25000],
    [40000, 3, 2, 1, 20000],
    [60000, 5, 3, 1, 18000],
    [80000, 4, 1, 0, 22000],
    # Add more data here if you have a larger dataset
]

# Separate features (X) and target (y) variables
X = [[car[0], car[1], car[2], car[3]] for car in dataset]
y = [car[4] for car in dataset]

# Create a Decision Tree Regressor
regressor = DecisionTreeRegressor()

# Train the model on the dataset
regressor.fit(X, y)

# Function to take input from the user for a new car
def get_new_car_input():
    mileage = int(input("Enter mileage (in miles): "))
    age = int(input("Enter age (in years): "))
    brand = int(input("Enter brand (e.g., 1 for Toyota, 2 for Honda, etc.): "))
    engine_type = int(input("Enter engine type (e.g., 0 for petrol, 1 for diesel, etc.): "))
    return [mileage, age, brand, engine_type]

# Get input for a new car
new_car = get_new_car_input()

# Use the trained model to predict the price of the new car
predicted_price = regressor.predict([new_car])[0]

# Display the predicted price
print(f"The predicted price of the new car is: ${predicted_price:.2f}")

# Display the decision path for the new car
tree_rules = export_text(regressor, feature_names=["mileage", "age", "brand", "engine type"])
print("Decision Path:")
print(tree_rules)
