# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('/content/dataset.csv')

# Data cleaning
df_cleaned = df.drop(columns=['Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21'])
df_cleaned['covid19_status'] = df_cleaned['covid19_status'].map({'positive': 1, 'negative': 0})
df_cleaned['gender'] = df_cleaned['gender'].map({'MALE': 1, 'FEMALE': 0})
df_cleaned['visit_concept'] = df_cleaned['visit_concept'].map({'Inpatient Visit': 1, 'Emergency Room Visit': 0})

# Handle missing values by filling NaN with the mode for categorical and mean for numeric columns
df_cleaned.fillna(df_cleaned.mode().iloc[0], inplace=True)

# Define features and target
X = df_cleaned.drop(columns=['to_patient_id', 'is_icu'])
y = df_cleaned['is_icu']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(),
    "SVM": SVC()
}

# Train and evaluate each model
results = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Train the model
    y_pred = model.predict(X_test_scaled)  # Make predictions

    accuracy = accuracy_score(y_test, y_pred)  # Accuracy
    report = classification_report(y_test, y_pred)  # Classification report
    cm = confusion_matrix(y_test, y_pred)  # Confusion matrix

    results[model_name] = {
        "Accuracy": accuracy,
        "Classification Report": report,
        "Confusion Matrix": cm
    }

# Display the results
for model_name, result in results.items():
    print(f"{model_name} - Accuracy: {result['Accuracy']}")
    print("Classification Report:")
    print(result["Classification Report"])
    print("Confusion Matrix:")
    print(result["Confusion Matrix"])
    print("\n")
