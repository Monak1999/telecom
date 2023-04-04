import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load data and train the model
data = pd.read_excel('C:/Users/Hp\Downloads/train.xlsx')

num_cols = data.select_dtypes(include=['float64', 'int64']).columns
print("Numerical columns:", num_cols)

# print categorical columns
cat_cols = data.select_dtypes(include=['object']).columns
print("Categorical columns:", cat_cols)

le = LabelEncoder()

# Convert all columns to strings
for col in data.columns:
    data[col] = data[col].astype(str)

# Apply the encoder to each column
for col in data.columns:
    data[col] = le.fit_transform(data[col])

# Encode categorical variables in a pandas DataFrame
for col in cat_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop(['Churn'], axis=1)
y = data['Churn']
clf = RandomForestClassifier()
clf.fit(X, y) 



# Save the model to a new pickle file
with open('D:\Final_Hackathon\churn.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Define the Streamlit app
def predict_churn(data):
    # Load the saved model from pickle file
    with open('D:\Final_Hackathon\churn.pkl', 'rb') as file:
        clf = pickle.load(file)

    # Make a prediction using the loaded model
    prediction = clf.predict(data)
    probability = clf.predict_proba(data)[:, 1]
    if prediction.any() == 1:
        return 'Churn', probability[0]
    else:
        return 'Not Churn', probability[0]
    return churn, probability 

# create a DataFrame from the input dictionary
    input_df = pd.DataFrame(input_dict)

def preprocess_data(df):
    
    # Drop unnecessary column
    df = df.drop('customerID', axis=1)
    data.drop('Churn', axis=1, inplace=True) 
    predictions = clf.predict(data)
    
    # Check if all columns in multi_cols are present in the dataframe
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    missing_cols = set(multi_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Encode multi-category features as binary columns
    # Create dummy variables for categorical variables with more than 2 categories
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    # Convert binary categorical variables to 0/1
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    df[binary_cols] = df[binary_cols].replace({'Yes': 1, 'No': 0})

    # Convert gender to binary
    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})

    # Scale numerical variables to have 0 mean and unit variance
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df



def main():
    # Set the app title
    st.title('Telecom Customer Churn Prediction')

    # Add a description
    st.write('This app predicts whether a telecom customer will churn or not based on their demographic and usage data.')

    # Add input fields for user data
    st.write('Enter customer data:')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (months)', 0, 100, 0)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
    device_protection = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
    tech_support = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
    streaming_tv = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=0.0, step=0.1)


    
    # Convert user data to a dataframe
    input_dict = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
    })
 


# Preprocess the data
data_processed = preprocess_data(data)

# Make predictions on the preprocessed data using the trained model
predictions = clf.predict(data)

# Make a prediction using the predict_churn function
result = clf.predict(data)
churn = result[0] # assigns the first value returned by the function call to the variable 'churn'
probability = result[1]

# Make the prediction and display the result
if st.button('Predict'):
   churn, probability = predict_churn(data)
   if churn == 0:
       st.write('The customer is not likely to churn.')
   else:
       st.write('The customer is likely to churn.')
# Display the prediction results

st.write('Churn: ', churn)
st.write('Probability: ', round(probability, 2)*100, '%')

if __name__=='__main__':
    main()
