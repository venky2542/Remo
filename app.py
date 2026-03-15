from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import joblib

app = Flask(__name__)
CORS(app) 
print("Loading NNDL Model & Data...")
model = tf.keras.models.load_model('nndl_churn_model.h5')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('model_features.pkl')
df = pd.read_csv('bank_customers_data.csv')

# Pre-calculate predictions for speed
X_data = df.drop(columns=['customerId', 'bankId', 'name', 'bankName', 'managerId', 'managerName', 'Churn'], errors='ignore')
X_data = pd.get_dummies(X_data, drop_first=True)

# Ensure columns match training exactly
X_data = X_data.reindex(columns=feature_cols, fill_value=0) 

X_scaled = scaler.transform(X_data)
df['Churn_Prob'] = model.predict(X_scaled).flatten() * 100

@app.route('/api/portal', methods=['GET'])
def get_portal():
    banks = df[['bankId', 'bankName', 'managerName']].drop_duplicates().to_dict('records')
    return jsonify({'banks': banks})

@app.route('/api/bank/<bank_id>', methods=['GET'])
def get_bank_data(bank_id):
    bank_df = df[df['bankId'] == bank_id]
    
    total = len(bank_df)
    at_risk = len(bank_df[bank_df['Churn_Prob'] > 60])
    
    top_risk = bank_df.sort_values(by='Churn_Prob', ascending=False).head(5)
    
    return jsonify({
        'bank_name': str(bank_df['bankName'].iloc[0]),
        'manager': str(bank_df['managerName'].iloc[0]),
        'total': int(total),
        'at_risk': int(at_risk),
        'safe': int(total - at_risk),
        'top_risk': top_risk[['customerId', 'name', 'Churn_Prob']].to_dict('records'),
        'all_customers': bank_df[['customerId', 'name', 'tenure', 'monthlyCharges', 'Churn_Prob']].to_dict('records')
    })

@app.route('/api/analyze/<cust_id>', methods=['GET'])
def analyze_customer(cust_id):
    # Find the specific customer
    cust = df[df['customerId'] == cust_id].iloc[0]
    prob = cust['Churn_Prob']
    
    # 🟢 THE FIX: Explicitly wrap the Pandas/NumPy data in standard Python int(), float(), and str()
    return jsonify({
        'id': str(cust['customerId']),
        'name': str(cust['name']),
        'contract': str(cust['contractType']),
        'billing': float(cust['monthlyCharges']),
        'tenure': int(cust['tenure']),
        'calls': int(cust['supportCalls']),
        'prob': float(round(prob, 1))
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)