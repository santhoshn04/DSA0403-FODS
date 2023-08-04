from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample dataset of past customers with their churn status and features
# Replace this with your own dataset
# Format: [usage minutes, contract duration, demographic data, churn status]
dataset = [
    [200, 12, 1, 0],   # Not churned (0)
    [100, 6, 2, 1],    # Churned (1)
    [300, 24, 3, 0],   # Not churned (0)
    [150, 18, 1, 1],   # Churned (1)
    # Add more data here if you have a larger dataset
]

# Separate features (X) and target (y) variables
X = [[customer[0], customer[1], customer[2]] for customer in dataset]
y = [customer[3] for customer in dataset]

# Create a Logistic Regression model
model = LogisticRegression()

# Scale the features (optional but recommended for logistic regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model on the dataset
model.fit(X_scaled, y)

# Function to take input from the user for a new customer
def get_new_customer_input():
    usage_minutes = float(input("Enter usage minutes: "))
    contract_duration = int(input("Enter contract duration (months): "))
    demographic_data = int(input("Enter demographic data (e.g., 1 for young, 2 for middle-aged, 3 for elderly): "))
    return [usage_minutes, contract_duration, demographic_data]

# Get input for a new customer
new_customer = get_new_customer_input()

# Scale the new customer's features (using the same scaler as before)
new_customer_scaled = scaler.transform([new_customer])

# Use the trained model to predict whether the new customer will churn or not
predicted_churn_status = model.predict(new_customer_scaled)

if predicted_churn_status[0] == 0:
    print("The model predicts that the customer will not churn.")
else:
    print("The model predicts that the customer will churn.")
