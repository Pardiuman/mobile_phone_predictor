import pandas as pd

# Load dataset with latin1 encoding
try:
    data = pd.read_csv("./data/mobile_data.csv", encoding="latin1")
except Exception as e:
    print(f"Failed to load dataset: {e}")
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
print(data.head())  # Prints all columns
