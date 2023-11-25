from flask import Flask, request, jsonify
import pickle  # Assuming you used joblib for saving the model, you can also use 'pickle' if that's what you used
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyngrok import ngrok

app = Flask(__name__)

# Load your trained machine learning model
model_path = 'C:/Users/unika/Documents/GitHub/BI_Project/models/xgb_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data, index=[0])

    # Perform any necessary preprocessing (e.g., scaling)
    scaler = StandardScaler()
    numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Handle categorical features
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming input data is in JSON format
        input_data = request.get_json()

        # Preprocess input data
        preprocessed_data = preprocess_input(input_data)

        # Make predictions using your machine learning model
        prediction = model.predict(preprocessed_data)

        # Return the result as JSON
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)



# from flask import *
# import json, time

# app = Flask(__name__)

# @app.route('/', methods=['GET'])
# def home_page():
#     data_set = {'Page': 'Home', 'Message': 'Successfully loaded the Home page', 'Timestamp': time.time()}
#     json_dump = json.dumps(data_set)

#     return json_dump



# @app.route('/user/', methods=['GET'])
# def request_page():
#     user_query = str(request.args.get('user'))

#     data_set = {'Page': 'Request', 'Message': f'Successfully got the request for {user_query}', 'Timestamp': time.time()}
#     json_dump = json.dumps(data_set)

#     return json_dump


# if __name__ == '__main__':
#     app.run(host="127.0.0.1", port=5000)