from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import joblib  # Import joblib for saving models

# # Load the dataset
# df = pd.read_csv("/Users/sanghvi/Desktop/SIH/dataset/synthetic_aluminium_data.csv")

# # Separate features and target variables
# y = df[['UTS', 'Elongation', 'Conductivity']]
# X = df[['CastingTemp', 'CoolingWaterTemp', 'CastingSpeed', 'EntryTempRollingMill', 
#         'EmulsionTemp', 'EmulsionPressure', 'EmulsionConcentration', 'RodQuenchWaterPressure']]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Initialize the Gradient Boosting model
# gb = GradientBoostingRegressor(random_state=42)

# # Wrap the model in MultiOutputRegressor for multi-output prediction
# multi_target_gb = MultiOutputRegressor(gb)

# # Fit the model on the training data
# multi_target_gb.fit(X_train, y_train)

# # Predict and evaluate on the training and testing set
# y_gb_train_pred = multi_target_gb.predict(X_train)
# y_gb_test_pred = multi_target_gb.predict(X_test)

# print("\nGradient Boosting Regressor (Multi-output)")
# print("MSE (test):", mean_squared_error(y_test, y_gb_test_pred))
# print("R^2 Score (test):", r2_score(y_test, y_gb_test_pred))
# print("MSE (train):", mean_squared_error(y_train, y_gb_train_pred))
# print("R^2 Score (train):", r2_score(y_train, y_gb_train_pred))

# # Hyperparameter tuning for Gradient Boosting using GridSearchCV
# param_grid_gb = {
#     'estimator__n_estimators': [50, 100, 150],
#     'estimator__max_depth': [3, 5, 7],
#     'estimator__learning_rate': [0.01, 0.1, 0.2],
#     'estimator__subsample': [0.8, 1.0]
# }

# grid_search_gb = GridSearchCV(estimator=multi_target_gb, param_grid=param_grid_gb, 
#                               cv=5, scoring='r2', n_jobs=-1)
# grid_search_gb.fit(X_train, y_train)

# # Best parameters for Gradient Boosting
# print("\nBest parameters from GridSearchCV for Gradient Boosting:", grid_search_gb.best_params_)

# # Evaluate with best estimator
# best_gb = grid_search_gb.best_estimator_
# y_best_gb_test_pred = best_gb.predict(X_test)

# print("\nOptimized Gradient Boosting Test MSE:", mean_squared_error(y_test, y_best_gb_test_pred))
# print("Optimized Gradient Boosting R^2 Score (test):", r2_score(y_test, y_best_gb_test_pred))

# # Save the scaler and best_gb model
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(best_gb, 'best_gb_model.pkl')

# print("\nScaler and Best Gradient Boosting model have been saved.")

# Function to load the saved models
def load_models():
    scaler = joblib.load(r"C:\Users\abhin\Desktop\PREDICTION\PREDICTION\Gradient Boosting Regressor\scaler.pkl")
    best_gb = joblib.load(r"C:\Users\abhin\Desktop\PREDICTION\PREDICTION\Gradient Boosting Regressor\best_gb_model.pkl")
    return scaler, best_gb

# Accept runtime input and make predictions
def get_runtime_input():
    print("\nEnter the input parameters for prediction:")
    casting_temp = float(input("Casting Temperature (°C): "))
    cooling_water_temp = float(input("Cooling Water Temperature (°C): "))
    casting_speed = float(input("Casting Speed (cm/s): "))
    entry_temp_rolling_mill = float(input("Entry Temperature at Rolling Mill (°C): "))
    emulsion_temp = float(input("Emulsion Temperature (°C): "))
    emulsion_pressure = float(input("Emulsion Pressure (bar): "))
    emulsion_concentration = float(input("Emulsion Concentration (%): "))
    rod_quench_water_pressure = float(input("Rod Quench Water Pressure (bar): "))
    
    # Construct input array
    input_features = [[casting_temp, cooling_water_temp, casting_speed, 
                       entry_temp_rolling_mill, emulsion_temp, emulsion_pressure, 
                       emulsion_concentration, rod_quench_water_pressure]]
    
    return input_features

# Main loop for making predictions
while True:
    # Load the saved models (uncomment the next line if you want to load models from disk)
    scaler, best_gb = load_models()
    
    # Get user input for prediction
    user_input = get_runtime_input()
    
    # Scale the user input using the saved scaler
    user_input_scaled = scaler.transform(user_input)
    
    # Make predictions using the loaded best_gb model
    predictions = best_gb.predict(user_input_scaled)
    
    print("\nPredicted Properties:")
    print(f"UTS (Ultimate Tensile Strength): {predictions[0][0]:.2f}")
    print(f"Elongation: {predictions[0][1]:.2f}")
    print(f"Conductivity: {predictions[0][2]:.2f}")
    
    cont = input("\nDo you want to make another prediction? (yes/no): ").strip().lower()
    if cont != 'yes':
        break
