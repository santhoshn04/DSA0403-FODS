                                                                                                                                                    from sklearn.linear_model import LinearRegression

                                                                                                                                                    # Sample dataset of houses with their respective prices
                                                                                                                                                    # You should replace this with your own dataset
                                                                                                                                                    # Format: [area, number of bedrooms, location, price]
                                                                                                                                                    dataset = [
                                                                                                                                                        [1500, 3, 1, 250000],
                                                                                                                                                        [2000, 4, 2, 350000],
                                                                                                                                                        [1200, 2, 1, 180000],
                                                                                                                                                        [1800, 3, 3, 280000],
                                                                                                                                                        # Add more data here if you have a larger dataset
                                                                                                                                                    ]

                                                                                                                                                    # Separate features (X) and target (y) variables
                                                                                                                                                    X = [[house[0], house[1], house[2]] for house in dataset]
                                                                                                                                                    y = [house[3] for house in dataset]

                                                                                                                                                    # Create a Linear Regression model
                                                                                                                                                    model = LinearRegression()

                                                                                                                                                    # Train the model on the dataset
                                                                                                                                                    model.fit(X, y)

                                                                                                                                                    # Function to take input from the user for a new house
                                                                                                                                                    def get_new_house_input():
                                                                                                                                                        area = float(input("Enter area (sq. ft.): "))
                                                                                                                                                        bedrooms = int(input("Enter number of bedrooms: "))
                                                                                                                                                        location = int(input("Enter location (e.g., 1 for city center, 2 for suburbs, etc.): "))
                                                                                                                                                        return [area, bedrooms, location]

                                                                                                                                                    # Get input for a new house
                                                                                                                                                    new_house = get_new_house_input()

                                                                                                                                                    # Use the trained model to predict the price of the new house
                                                                                                                                                    predicted_price = model.predict([new_house])

                                                                                                                                                    print(f"The predicted price of the new house is: ${predicted_price[0]:.2f}")

