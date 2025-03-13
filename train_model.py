import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import argparse

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mobile_data", type=str, required=True, help="Path to mobile data")
args = parser.parse_args()

# Since Azure ML mounts the input as a directory, append the filename
input_dir = args.mobile_data  # This is the mounted directory, e.g., /mnt/azureml/.../INPUT_mobile_data
csv_file_path = os.path.join(input_dir, "mobile_data.csv")  # Explicitly specify the file

# Load dataset with latin1 encoding
try:
    data = pd.read_csv(csv_file_path, encoding="latin1")
except Exception as e:
    print(f"Failed to load dataset from {csv_file_path}: {e}")
    raise

# Debug: Print raw data
print("Raw data sample:")
print(data.head())

# Preprocessing
try:
    # Clean Mobile Weight: Remove 'g'
    data['Mobile Weight'] = data['Mobile Weight'].str.replace('g', '').astype(float)
    # Clean RAM: Remove 'GB', handle '12MP / 4K' by taking first number
    data['RAM'] = data['RAM'].str.replace('GB', '').str.split('/').str[0].str.extract('(\d+)').astype(float)
    # Clean Battery Capacity: Remove ',' and 'mAh'
    data['Battery Capacity'] = data['Battery Capacity'].str.replace(',', '').str.replace('mAh', '').astype(float)
    # Clean Screen Size: Remove 'inches', extract first number for formats like '6.7  (main), 2.7  (external)'
    data['Screen Size'] = data['Screen Size'].str.replace('inches', '').str.extract('(\d+\.\d+|\d+)').astype(float)
    # Clean Launched Year: Already numeric
    data['Launched Year'] = data['Launched Year'].astype(int)
    # Clean Launched Price (USA): Remove 'USD ' and ','
    data['Launched Price (USA)'] = data['Launched Price (USA)'].str.replace('USD ', '').str.replace(',', '').astype(float)
    # Drop rows with NaN in features or target
    data = data.dropna(subset=['Mobile Weight', 'RAM', 'Battery Capacity', 'Screen Size', 'Launched Year', 'Launched Price (USA)'])
except Exception as e:
    print(f"Error in preprocessing: {e}")
    raise

# Debug: Print preprocessed data with all columns
print("Preprocessed data sample:")
print(data.head())

# Selecting relevant features
X = data[['Mobile Weight', 'RAM', 'Battery Capacity', 'Screen Size', 'Launched Year']]
y = data['Launched Price (USA)']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Save the model
output_dir = os.path.join(os.environ.get("AZUREML_OUTPUTS_DEFAULT", "."), "outputs")
print(f"Output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "mobile_price_predictor.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
