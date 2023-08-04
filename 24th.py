import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Sample data (replace this with your actual dataset)
# Assuming X contains symptom features and y contains the labels (0 or 1)
X = np.array([[symptom1_value, symptom2_value, symptom3_value],   # Replace symptom1_value, symptom2_value, etc., with actual values
              [symptom1_value, symptom2_value, symptom3_value],
              ...
              [symptom1_value, symptom2_value, symptom3_value]])

y = np.array([label1, label2, ..., labelN])  # Replace label1, label2, etc., with actual labels (0 or 1)

def predict_condition(new_patient_features, k):
    # Create the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the KNN classifier on the training data
    knn_classifier.fit(X_train, y_train)

    # Predict the condition for the new patient
    prediction = knn_classifier.predict([new_patient_features])

    return prediction[0]

def main():
    # Ask the user for the features of the new patient
    new_patient_features = []
    num_features = len(X[0])  # Assuming all patients have the same number of features

    for i in range(num_features):
        feature_value = float(input(f"Enter feature {i+1}: "))
        new_patient_features.append(feature_value)

    # Ask the user for the value of k (number of neighbors)
    k = int(input("Enter the value of k (number of neighbors): "))

    # Predict the condition for the new patient
    prediction = predict_condition(new_patient_features, k)

    if prediction == 0:
        print("The patient does not have the medical condition.")
    else:
        print("The patient has the medical condition.")

if __name__ == "__main__":
    main()
