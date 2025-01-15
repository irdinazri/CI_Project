import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# Define Gaussian Membership Function
def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


# Function to simulate dataset if file is not available
def create_sample_dataset(file_path):
    data = {
        "Age": np.random.randint(20, 80, 1000),
        "Sex": np.random.choice(["Male", "Female"], 1000),
        "Cholesterol": np.random.randint(150, 300, 1000),
        "Blood Pressure": [f"{np.random.randint(100, 140)}/{np.random.randint(60, 90)}" for _ in range(1000)],
        "Heart Rate": np.random.randint(60, 120, 1000),
        "Diabetes": np.random.choice(["Yes", "No"], 1000),
        "Diet": np.random.choice(["Healthy", "Average", "Unhealthy"], 1000),
        "Heart Attack Risk": np.random.choice([0, 1], 1000)
    }
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False, sheet_name="heart_attack_prediction_dataset")


# File path to the dataset
file_path = "heart_attack_prediction_dataset.xlsx"

# Check if file exists; if not, create a sample dataset
try:
    data = pd.ExcelFile(file_path).parse('heart_attack_prediction_dataset')
except FileNotFoundError:
    create_sample_dataset(file_path)
    data = pd.ExcelFile(file_path).parse('heart_attack_prediction_dataset')

# Preprocess data
columns = ['Age', 'Sex', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Diabetes', 'Diet', 'Heart Attack Risk']
data_filtered = data[columns].copy()
data_filtered['Sex'] = data_filtered['Sex'].map({"Male": 0, "Female": 1})
data_filtered['Diabetes'] = data_filtered['Diabetes'].map({"Yes": 1, "No": 0})
data_filtered['Diet'] = data_filtered['Diet'].map({"Healthy": 0, "Average": 1, "Unhealthy": 2})


# Convert Blood Pressure from 'systolic/diastolic' to systolic values
def extract_systolic(bp):
    try:
        systolic = int(bp.split('/')[0])
        return systolic
    except:
        return np.nan


data_filtered['Blood Pressure'] = data_filtered['Blood Pressure'].apply(extract_systolic)
data_filtered.dropna(subset=['Blood Pressure'], inplace=True)

# Define features and target
X = data_filtered[['Age', 'Sex', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Diabetes', 'Diet']]
y = data_filtered['Heart Attack Risk']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Define membership functions
def define_membership_functions():
    return {
        'Age': [
            ('young', gaussmf, {'mean': 25, 'sigma': 10}),
            ('middle-aged', gaussmf, {'mean': 45, 'sigma': 10}),
            ('old', gaussmf, {'mean': 65, 'sigma': 10}),
        ],
        'Cholesterol': [
            ('low', gaussmf, {'mean': 150, 'sigma': 30}),
            ('normal', gaussmf, {'mean': 200, 'sigma': 30}),
            ('high', gaussmf, {'mean': 250, 'sigma': 30}),
        ],
        'Blood Pressure': [
            ('low', gaussmf, {'mean': 110, 'sigma': 10}),
            ('normal', gaussmf, {'mean': 120, 'sigma': 10}),
            ('high', gaussmf, {'mean': 140, 'sigma': 10}),
        ],
        'Heart Rate': [
            ('low', gaussmf, {'mean': 60, 'sigma': 10}),
            ('normal', gaussmf, {'mean': 80, 'sigma': 10}),
            ('high', gaussmf, {'mean': 100, 'sigma': 10}),
        ],
        'Diet': [
            ('healthy', gaussmf, {'mean': 0, 'sigma': 1}),
            ('average', gaussmf, {'mean': 1, 'sigma': 1}),
            ('unhealthy', gaussmf, {'mean': 2, 'sigma': 1}),
        ],
    }


# Placeholder for ANFIS model (to be implemented with custom logic or an alternative library)
class ANFISModel:
    def __init__(self, membership_functions):
        self.membership_functions = membership_functions

    def fit(self, X, y, epochs=50, learning_rate=0.01):
        print("Training ANFIS model (placeholder)...")
        # Add your custom training logic here

    def predict(self, X):
        print("Predicting with ANFIS model (placeholder)...")
        return np.random.choice([0, 1], size=X.shape[0])  # Placeholder logic


membership_functions = define_membership_functions()
model = ANFISModel(membership_functions)

# Fit the ANFIS model
try:
    model.fit(X_scaled, y, epochs=50, learning_rate=0.01)
except Exception as e:
    print(f"Error during training: {e}")


# GUI Implementation
class HeartAttackPredictionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Heart Risk Prediction")
        self.root.geometry("700x600")
        self.root.configure(bg="#1e1e2f")

        # Title
        tk.Label(self.root, text="Heart Risk Prediction", font=('Futura', 18, 'bold'), bg="#1e1e2f", fg="#ffffff").pack(
            pady=10)

        self.label = tk.Label(self.root, text="'Empowering your health with precision predictions'",
                              font=('Futura', 12, 'italic'), bg="#1e1e2f", fg="#c0c0c0")  # Light gray text
        self.label.pack(padx=20, pady=5)

        tk.Button(self.root, text="Let's Get Started!", font=('Futura', 14), bg="#4caf50", fg="#ffffff",
                  command=self.start).pack(pady=20)
        self.root.mainloop()

    def start(self):
        start_window = tk.Toplevel(self.root)
        start_window.title("Prediction")
        start_window.geometry("700x600")
        start_window.configure(bg="#1e1e2f")

        tk.Label(start_window, text="Fill out the following details:", font=('Futura', 16), bg="#1e1e2f",
                 fg="#ffffff").pack(pady=10)

        # Define form fields
        fields = ["Name", "Age", "Sex (Male/Female)", "Cholesterol", "Blood Pressure", "Heart Rate",
                  "Diabetes (Yes/No)", "Diet"]
        self.entries = {}

        for idx, field in enumerate(fields):
            tk.Label(start_window, text=field, bg="#1e1e2f", fg="#ffffff").pack(anchor='w', padx=20)
            entry = tk.Entry(start_window)
            entry.pack(pady=5, padx=20, anchor='w')
            self.entries[field] = entry

        # Predict button
        tk.Button(start_window, text="Predict", command=self.predict).pack(pady=20)

    def predict(self):
        try:
            # Collect inputs
            age = int(self.entries["Age"].get())
            gender_input = self.entries["Sex (Male/Female)"].get().strip().lower()
            if gender_input == 'male':
                gender = 0
            elif gender_input == 'female':
                gender = 1
            else:
                raise ValueError("Invalid gender. Please enter Male or Female.")

            cholesterol = float(self.entries["Cholesterol"].get())

            # Process Blood Pressure
            blood_pressure_input = self.entries["Blood Pressure"].get().strip()
            blood_pressure = extract_systolic(blood_pressure_input)
            if np.isnan(blood_pressure):
                raise ValueError("Invalid blood pressure format. Use systolic/diastolic (e.g., 120/80).")

            heart_rate = float(self.entries["Heart Rate"].get())
            diabetes = 1 if self.entries["Diabetes (Yes/No)"].get().strip().lower() == 'yes' else 0

            # Process Diet
            diet_input = self.entries["Diet"].get().strip().capitalize()
            diet_mapping = {"Healthy": 0, "Average": 1, "Unhealthy": 2}
            if diet_input in diet_mapping:
                diet = diet_mapping[diet_input]
            else:
                raise ValueError("Invalid diet. Please select Healthy, Average, or Unhealthy.")

            # Scale input
            input_data = scaler.transform([[age, gender, cholesterol, blood_pressure, heart_rate, diabetes, diet]])

            # Predict using ANFIS (placeholder prediction)
            prediction = model.predict(input_data)[0]
            result = "HIGH RISK" if prediction == 1 else "LOW RISK"

            # Display result
            messagebox.showinfo("Prediction Result", f"Predicted Heart Risk: {result}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


HeartAttackPredictionGUI()


