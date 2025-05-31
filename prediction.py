import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = joblib.load('model/kmeans_clustering_model.joblib')
scaler = joblib.load('model/scaler.joblib')

def encoding_data(df):
    transAtribut = ["BusinessTravel", "Department", "EducationField", "JobRole", "OverTime"]
    for trans in transAtribut:
        labelEncoding = joblib.load(f'model/labelEncoding_{trans}.joblib')
        df[trans] = labelEncoding.transform(df[trans])

    return df

data = {
    'Age': [30, 40, 50],
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'DistanceFromHome': [10, 20, 15],
    'Education': [2, 3, 4],
    'EducationField': ['Life Sciences', 'Medical', 'Marketing'],
    'EnvironmentSatisfaction': [3, 2, 4],
    'JobInvolvement': [2, 3, 4],
    'JobLevel': [2, 2, 3],
    'JobRole': ['Research Scientist', 'Research Director', 'Sales Executive'],
    'JobSatisfaction': [4, 3, 2],
    'MonthlyIncome':[6500, 13237, 6000],
    'MonthlyRate': [13430, 20978, 10877],
    'NumCompaniesWorked': [3, 2, 5],
    'OverTime': ['Yes', 'Yes', 'Yes'],
    'PercentSalaryHike': [10, 15, 13],
    'PerformanceRating': [3, 4, 3],
    'RelationshipSatisfaction': [3, 3, 3],
    'StockOptionLevel': [1, 3, 1],
    'TotalWorkingYears': [5, 10, 5],
    'WorkLifeBalance': [3, 4, 2],
    'YearsAtCompany': [5, 10, 10],
    'YearsInCurrentRole': [2, 5, 8],
    'YearsSinceLastPromotion': [1, 3, 2],
    'YearsWithCurrManager': [2, 3, 3]
}

df = pd.DataFrame(data)

df_encoded = encoding_data(df)

df_scaled = scaler.transform(df_encoded)

predictions = model.predict(df_scaled)
predictions = predictions.tolist()

for i in predictions:
    print("Potentially Attrition" if i == 0 else "No Attrition")