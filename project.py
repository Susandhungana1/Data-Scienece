import pandas as pd
from sklearn.model_selection import train_test_split
import sweetviz as sv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('healthcare_dataset.csv')
print(data.head())
print(data.shape)

data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
print(data.columns)
print(data.isnull().sum())
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
print(data.shape)

report = sv.analyze(data)
report.show_html('report.html')

data['Age_Group'] = pd.cut(data['age'], bins=[0, 18, 60, 120], labels=["Child", "Adult", "Senior"], right=False)
data = pd.get_dummies(data, columns=['Age_Group','gender', 'blood_type', 'admission_type', 'medical_condition'], drop_first=True)
print(data.columns)

X = data.drop(columns=["name", "discharge_date", "test_results", "medication"])
Y = LabelEncoder().fit_transform(data['test_results'])

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)

age = int(input("Enter Age: \n"))
gender = input("Enter Gender (Male/Female): \n")
blood_type = input("Enter Blood Type (A+/A-/B+/B-/O+/O-): \n")
medical_condition = input("Enter Medical Condition: \n")
admission_type = input("Enter Admission Type (Urgent/Emergency/Elective): \n")

input_data = [[age, gender, blood_type, medical_condition, admission_type]]
input_data = pd.DataFrame(input_data, columns=['age', 'gender', 'blood_type', 'medical_condition', 'admission_type'])
input_data['Age_Group'] = pd.cut(input_data['age'], bins=[0, 18, 60, 120], labels=["Child", "Adult", "Senior"], right=False)
input_data = pd.get_dummies(input_data, columns=['gender','medical_condition', 'blood_type', 'admission_type'], drop_first=True)
input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

if input_data.isnull().values.any():
    print("Input data is missing values")
else:
    input_data = input_data.values
    y_pred = model.predict(input_data)
    y_pred = LabelEncoder().fit(data['test_results']).inverse_transform(y_pred)
    print("Prediction:", y_pred[0])
