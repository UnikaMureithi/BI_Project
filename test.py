import requests
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_prediction(input_data):
    # Send a POST request to the Flask API
    url = "https://36e7-41-80-116-75.ngrok-free.app/predict"
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

def preprocess_input(input_data):
    # Convert input data to a dictionary if not already
    if not isinstance(input_data, dict):
        input_data = {
            "age": input_data[0],
            "height": input_data[1],
            "weight": input_data[2],
            "gender": input_data[3],
            "ap_hi": input_data[4],
            "ap_lo": input_data[5],
            "cholesterol": input_data[6],
            "gluc": input_data[7],
            "smoke": input_data[8],
            "alco": input_data[9],
            "active": input_data[10]
        }

    # Perform any necessary preprocessing (scaling, one-hot encoding, etc.)
    scaler = StandardScaler()
    numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    input_data[numerical_features] = scaler.fit_transform(pd.DataFrame(input_data, index=[0])[numerical_features])

    # Handle categorical features if any
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    input_data = pd.get_dummies(pd.DataFrame(input_data, index=[0]), columns=categorical_features, drop_first=True)

    return input_data

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
