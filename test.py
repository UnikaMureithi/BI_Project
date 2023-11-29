import requests
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_input(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])

    # Define numerical features for scaling
    numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

    # Initialize scaler
    scaler = StandardScaler()

    # Scale numerical features
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Handle categorical features
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    df[categorical_features] = df[categorical_features].astype(str)  # Convert to string before one-hot encoding

    # Ensure that all expected columns are present
    expected_columns = numerical_features + categorical_features
    missing_columns = set(expected_columns) - set(df.columns)

    # Add missing columns with dummy values
    for column in missing_columns:
        df[column] = 0

    # Reorder columns to match the model's expected order
    df = df[expected_columns]

    # Convert DataFrame to dictionary
    input_dict = df.to_dict(orient='records')[0]

    return input_dict



def get_prediction(input_data):
    # Send a POST request to the Flask API
    url = "https://7d26-197-139-46-5.ngrok-free.app/predict"
    headers = {"Content-Type": "application/json"}
    
    try:
        # Preprocess input data
        preprocessed_data = preprocess_input(input_data)

        # Send the preprocessed data as JSON
        response = requests.post(url, headers=headers, data=json.dumps(preprocessed_data))
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        # Try to parse the response as JSON and return it
        return response.json()

    except requests.RequestException as e:
        # Handle any request exception (e.g., connection error)
        return {"error": f"Request error: {str(e)}"}

    except json.JSONDecodeError:
        # Handle JSON decoding error
        return {"error": "Could not parse response as JSON"}


# if __name__ == "__main__":
#     # Example user input
#     user_input = {
#         "age": 40,
#         "height": 160,
#         "weight": 70,
#         "gender": 1,
#         "ap_hi": 120,
#         "ap_lo": 80,
#         "cholesterol": 1,
#         "gluc": 1,
#         "smoke": 0,
#         "alco": 0,
#         "active": 1
#     }

#     # Get the prediction
#     prediction = get_prediction(user_input)

#     # Print the prediction
#     print("Prediction:", prediction)



if __name__ == "__main__":
    # Ask the user for input data
    age = int(input("Enter age: "))
    height = int(input("Enter height: "))
    weight = float(input("Enter weight: "))
    gender = int(input("Enter gender (1 for male, 2 for female): "))
    ap_hi = int(input("Enter systolic blood pressure (ap_hi): "))
    ap_lo = int(input("Enter diastolic blood pressure (ap_lo): "))
    cholesterol = int(input("Enter cholesterol level (1: normal, 2: above normal, 3: well above normal): "))
    gluc = int(input("Enter glucose level (1: normal, 2: above normal, 3: well above normal): "))
    smoke = int(input("Enter smoking status (0 for non-smoker, 1 for smoker): "))
    alco = int(input("Enter alcohol intake status (0 for non-drinker, 1 for drinker): "))
    active = int(input("Enter physical activity status (0 for inactive, 1 for active): "))

    # Convert user input to a dictionary
    user_input = {
        "age": age,
        "height": height,
        "weight": weight,
        "gender": gender,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }

    # Get the prediction
    prediction = get_prediction(user_input)

    # Print the prediction
    print("Prediction:", prediction)
