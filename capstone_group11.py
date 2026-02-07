import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report


# --- Function 1: Load Data ---
def load_flight_data(path_or_url):
    """
    Load dataset from a local path or a URL.
    """
    # 1. 检查是否是网络链接 (以 http 开头)
    if str(path_or_url).startswith(('http://', 'https://')):
        try:
            print(f"Loading data from URL: {path_or_url}")
            return pd.read_csv(path_or_url)
        except Exception as e:
            print(f"Error loading from URL: {e}")
            return None

    # 2. 如果是本地路径，保留你原有的逻辑
    # If a folder is passed
    if os.path.isdir(path_or_url):
        files = os.listdir(path_or_url)
        csv_file = next((f for f in files if f.endswith('.csv')), None)
        if csv_file:
            full_path = os.path.join(path_or_url, csv_file)
            print(f"Loading file from folder: {csv_file}")
            return pd.read_csv(full_path)

    # If a direct file path is passed
    elif os.path.isfile(path_or_url):
        print(f"Loading specific file: {path_or_url}")
        return pd.read_csv(path_or_url)

    print("Error: No CSV file found at the given path or URL.")
    return None


# --- Function 2: Clean Column Names ---
def clean_column_names(df):
    """
    Standardize column names to lowercase and handle spaces.
    """
    df.columns = df.columns.str.strip()
    rename_map = {
        'Departure Date & Time': 'dep_time',
        'Arrival Date & Time': 'arr_time',
        'Total Fare (BDT)': 'price',
        'Day': 'journey_day'
    }
    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


# --- Function 3: Process Date Features ---
def process_date_features(df):
    """
    Extract temporal features from datetime columns.
    """
    df['dep_time'] = pd.to_datetime(df['dep_time'])
    df['dep_month'] = df['dep_time'].dt.month
    df['dep_hour'] = df['dep_time'].dt.hour
    df['day_of_week'] = df['dep_time'].dt.dayofweek  # 0=Monday, 6=Sunday
    return df


# --- Function 4: Data Preparation (Encoding & Splitting) ---
def prepare_data_for_model(df, target_col='price', task='regression'):
    """
    Prepares data for modeling:
    1. Drops ID-like or redundant columns.
    2. One-Hot Encodes categorical features.
    3. Splits into Train/Test sets.

    Args:
    - task: 'regression' (predict price) or 'classification' (predict cabin class)
    """
    # 1. Select features
    # Drop derived or leaky columns (like base_fare which is part of price)
    cols_to_drop = ['dep_time', 'arr_time', 'source_name', 'destination_name',
                    'base_fare_(bdt)', 'tax_&_surcharge_(bdt)']

    # If doing regression, we predict price, so drop it from X.
    # If doing classification, we predict class, so drop it from X (and maybe price if we want).

    data = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    if task == 'regression':
        X = data.drop(columns=[target_col], errors='ignore')
        y = df[target_col]
    else:  # classification (predicting 'class')
        X = data.drop(columns=[target_col], errors='ignore')
        y = df[target_col]

    # 2. One-Hot Encoding for categorical variables
    # (Pandas get_dummies is smarter than sklearn OneHotEncoder for quick analysis)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 3. Train Test Split
    print(f"Data prepared for {task}. Feature shape: {X_encoded.shape}")
    return train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# --- Function 5: Train Regression Models (Linear & Random Forest) ---
def train_regression_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates Linear Regression and Random Forest.
    """
    results = {}

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['Linear Regression'] = {
        'R2': r2_score(y_test, y_pred_lr),
        'RMSE': mean_squared_error(y_test, y_pred_lr) ** 0.5,
        'model': lr
    }

    # Model 2: Random Forest (Limited depth to save time on large data)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        'R2': r2_score(y_test, y_pred_rf),
        'RMSE': mean_squared_error(y_test, y_pred_rf) ** 0.5,
        'model': rf
    }

    return results


# --- Function 6: Train Classification Model (Decision Tree) ---
def train_classification_model(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates Decision Tree Classifier.
    """
    # Model 3: Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Decision Tree Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return dt, cm
