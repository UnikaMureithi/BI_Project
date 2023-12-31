# BI_Project
The training data is the largest portion of from the dataset and was used to train the machine learning model. It contains historical data, including the features (independent variables) and the target variable (CVD status in this case). The machine learning model learns patterns and relationships from this data to make predictions. It adjusts its internal parameters during training to minimize errors or discrepancies between its predictions and the actual target values.

In this project, the training data consists of a subset of the dataset, excluding a portion that will be used for testing and validation. After training the model on the training data, the validation data was used to assess its performance and tune hyperparameters. This helps to prevent overfitting (a model fitting the training data too closely and performing poorly on new data).

In this project, the validation data would help to optimize the model's performance and make decisions about the choice of algorithms, feature engineering, and hyperparameter tuning. The testing data should represent data that the model has never seen during training or validation, making it a reliable indicator of how well the model will perform in real-world scenarios. The testing data provided an estimate of the model's predictive accuracy and its ability to assess the risk of CVDs on unseen data, which is crucial for reliability.

#### 136017_Prediction_Model.ipynb (Jupyter Notebook File):
This file can easily be viewed as a markdown. The contents of this notebook show a clear breakdown of how each of the Labs learnt in class was applied specifically to this project, as aligned with knowledge discovery in databases (KDD).
* Data Selection
* Data Cleansing and Pre-Processing
* Exploratory Data Analysis
* Data Transformation
* Selection of the Data Mining Algorithm
* Utilization of the Data Mining Algorithm
* Model Consolidation

### Model Consolidation Using Ngrok
To make the trained model accessible for predictions, the Flask API has been deployed using Ngrok for tunneling. The consolidated model can be accessed using the following URL: https://e135-197-139-46-5.ngrok-free.app -> http://localhost:5000

### To test the Model:
Utulize the ```test.py``` file to test the deployed model. Run the script and follow along with the promts whereby one is supposed to input the reuired data. The script will send a POST request to the deployed API and print the prediction of Absence or Presence of CVD.
Run this command:
```python test.py```
