import os
import pandas as pd
# import matplotlib.pyplot as plt
import joblib
import pickle
import json
import numpy as np
from django.shortcuts import render

def dashboard_home(request):
    # Read file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    balanced_path = os.path.join(BASE_DIR, 'dashboard', 'data', 'balanced_df.csv')
    balanced_df = pd.read_csv(balanced_path)
    csv_path = os.path.join(BASE_DIR, 'dashboard', 'data', 'cleaned_df.csv')

    cleaned_df = pd.read_csv(csv_path)

    avg_expenditure = int(cleaned_df['EXPTOT'].mean())

    ins_type_form = "Private Only"  # default type for initial view
    target_insurance_type_enc = lambda x:(
        'Private Only' if x == 0 else
        'Public Only'  if x == 1 else
        'Uninsured' if x == 2 else 3
    )

    filtered_df = cleaned_df[cleaned_df['INSURANCE_TYPE'] == ins_type_form].copy()
    
    insurance_type_encoded = (
    0 if ins_type_form == 'Private Only' else
    1 if ins_type_form == 'Public Only' else
    2 if ins_type_form == 'Uninsured' else 3 )  

    cleaned_filtered_df = balanced_df[balanced_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded].copy()

    if not filtered_df.empty:
        avg_charges_per_head = filtered_df['CHGTOT'].mean()
    else:
        avg_charges_per_head = 0

    # OOP Burden
    avg_oop_burden = round(filtered_df['OOP_BURDEN'].mean() * 100, 2)

    # Catastrophic OOP Rate
    catastrophic_rate = round(filtered_df['CATASTROPHIC_OOP'].mean() * 100, 2)

    # Health label mapping
    health_labels = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
    # avg health score based on insurance type
    avg_health_score = cleaned_filtered_df['HEALTH'].count() 
    # Create dict for all insurance types
    all_health_data = {}
    for ins_type in cleaned_df['INSURANCE_TYPE'].unique():
        df = cleaned_df.loc[cleaned_df['INSURANCE_TYPE'] == ins_type].copy()
        df.loc[:, 'HEALTH_LABEL'] = df['HEALTH'].map(health_labels)
        counts = df['HEALTH_LABEL'].value_counts().reindex(health_labels.values(), fill_value=0).to_dict()
        all_health_data[ins_type] = counts

    # Map short name for display
    if ins_type_form == "Private Only":
        ins_type_display = 'Private'
    elif ins_type_form == "Public Only":
        ins_type_display = 'Public'
    elif ins_type_form == "Mixed":
        ins_type_display = 'Mixed'
    else:
        ins_type_display = 'Uninsured'
    
    avg_health_df = (
    cleaned_filtered_df.groupby('INSURANCE_TYPE_ENC')['HEALTH']
    .mean()
    .reset_index())

    if not avg_health_df.empty:
        match = avg_health_df.loc[
            avg_health_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded, 'HEALTH'
        ]
        avg_health_score = float(match.values[0]) if not match.empty else 0.0
    else:
        avg_health_score = 0.0

    prediction_prob =0
    predicted_health_status= 0
    suitability_status=None

    
    

   
     # Prediction handling
    prediction = None
    if request.method == 'POST':
        # Collect form data
        sex = 1 if request.POST.get('SEX') == 'Male' else 2
        marstat = 1 if request.POST.get('MARSTAT') == 'Single' else 2
        cancer = 1 if request.POST.get('CANCEREV') == 'Yes' else 0
        chol = 1 if request.POST.get('CHOLHIGHEV') == 'Yes' else 0
        diabetes = 1 if request.POST.get('DIABETICEV') == 'Yes' else 0
        heart = 1 if request.POST.get('HEARTCONEV') == 'Yes' else 0
        hypertension = 1 if request.POST.get('HYPERTEN') == 'Yes' else 0
        hypertension_age = int(request.POST.get('HYPERTENAGE', 0))
        age = int(request.POST.get('AGE', 0))
        ins_type_form = request.POST.get('INSURANCE_TYPE')
        family_income = float(request.POST.get('FAMILY_INCOME', 0))
        family_size = int(request.POST.get('FAMILY_SIZE', 1))
        # target_insurance_type = request.POST.get('INSURANCE_TYPE')
        # Encode insurance type
        ins_map = {'Private Only': 0, 'Public Only': 1, 'Uninsured': 2, 'Mixed':3}
        ins_encoded = ins_map.get(ins_type_form, 2)

        insurance_type_encoded = (
            0 if ins_type_form == 'Private Only' else
            1 if ins_type_form == 'Public Only' else
            2 if ins_type_form == 'Uninsured' else 3 )  

        filtered_df = cleaned_df[cleaned_df['INSURANCE_TYPE'] == ins_type_form].copy()
       
         # OOP Burden
        avg_oop_burden = round(filtered_df['OOP_BURDEN'].mean() * 100, 2)

        # Catastrophic OOP Rate
        catastrophic_rate = round(filtered_df['CATASTROPHIC_OOP'].mean() * 100, 2)
        
        
        # AVG Charges/head
        if not filtered_df.empty:
                avg_charges_per_head = filtered_df['CHGTOT'].mean()
        else:
            avg_charges_per_head = 0

        # avg heaklth score
        cleaned_filtered_df = balanced_df[balanced_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded].copy()
        avg_health_df = (
    cleaned_filtered_df.groupby('INSURANCE_TYPE_ENC')['HEALTH']
    .mean()
    .reset_index())

        if not avg_health_df.empty:
            match = avg_health_df.loc[
                avg_health_df['INSURANCE_TYPE_ENC'] == insurance_type_encoded, 'HEALTH'
            ]
            avg_health_score = float(match.values[0]) if not match.empty else 0.0
        else:
            avg_health_score = 0.0
        

        # Prepare features for model
        features = np.array([[sex, marstat, family_size, family_income, age, cancer, chol,
                              diabetes, heart, hypertension, hypertension_age, ins_encoded]])
        

        if ins_type_form == "Private Only":
            ins_type_display = 'Private'
        elif ins_type_form == "Public Only":
            ins_type_display = 'Public'
        elif ins_type_form == "Mixed":
            ins_type_display = 'Mixed'
        else:
            ins_type_display = 'Uninsured'
        
        # Predict probability
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, 'dashboard', 'model', 'my_trained_model.pkl')
        model = pickle.load(open(MODEL_PATH, 'rb'))

        prediction = model.predict_proba(features)[0][1] * 100
        # Make prediction
        prediction = model.predict(features)[0]  # 0 = bad, 1 = good
        prediction_prob = model.predict_proba(features)[0][prediction] * 100
        predicted_health_status = "Good" if prediction == 1 else "Bad"
        suitability_status = "Suitable" if predicted_health_status == "Good" else "Not Suitable"
        print(f"prediction_prob = {prediction_prob}\n avg_charges_per_head = {avg_charges_per_head  }")


    context = {
        'avg_expenditure': avg_expenditure,
        'insurance_type': ins_type_display,
        'avg_charges_per_head': f"${avg_charges_per_head:,.2f}",
        "avg_oop_burden": avg_oop_burden,
        "catastrophic_rate": catastrophic_rate,
        'health_status_counts': all_health_data[ins_type_form],
        'all_health_data': json.dumps(all_health_data),
        'insurance_types': list(all_health_data.keys()),
        'avg_health_score':avg_health_score,
        "prediction": prediction_prob,
        "predicted_health_status": predicted_health_status,
        "predicted_insurance_type": request.POST.get("INSURANCE_TYPE") if request.method == "POST" else None,
        "suitability_status" : suitability_status
   
    }

    return render(request, "index.html", context)

