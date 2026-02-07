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


# --- Data Extraction ---
def load_flight_data(url):
    """
    Load dataset from a URL.
    """
    try:
        print(f"Loading data from URL: {url}")
        df = pd.read_csv(url)

        if df is not None and not df.empty:
            print(f"✅ Success: Data loaded! Shape: {df.shape}")
            return df
        else:
            print("⚠️ Warning: Data loaded but the dataset is empty.")
            return None

    except Exception as e:
        print(f"Error loading from URL: {e}")
        return None

# --- Data Cleanup 1: Clean Column Names ---
def clean_column_names(df):
    """
    Standardize column names to lowercase and handle spaces.
    """
    # 1. Remove whitespace from the beginning and end of column names
    df.columns = df.columns.str.strip()
    
    # 2. Define a dictionary mapping old names to new names
    rename_map = {
        'Departure Date & Time': 'dep_time',
        'Arrival Date & Time': 'arr_time',
        'Total Fare (BDT)': 'price',
        'Day': 'journey_day'
    }
    
    # 3. Execute the renaming process
    df = df.rename(columns=rename_map)
    
    # 4. Convert all column names to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    print("Column cleaning complete! Current columns:", list(df.columns))
    return df

# --- Data Cleanup 2: Process Date Features ---
def process_date_features(df):
    """
    Extract temporal features from datetime columns.
    """
    # Ensure 'dep_time' is in datetime format
    df['dep_time'] = pd.to_datetime(df['dep_time'])
    
    # Extract the month
    df['dep_month'] = df['dep_time'].dt.month
    
    # Extract the hour and day of week
    df['dep_hour'] = df['dep_time'].dt.hour
    df['day_of_week'] = df['dep_time'].dt.dayofweek  # 0=Monday, 6=Sunday
    print("Date feature processing complete! Added 'dep_month' and 'dep_hour'.")
    return df



# --- Visualization 4: Plot Correlation Heatmap ---
# This satisfies the requirement: Data Visualization
def plot_correlation_heatmap(df):
    """
    Plots a heatmap to show correlation between numerical variables.
    """
    # 1. Select only numerical columns (Correlation only works with numbers)
    numerical_df = df.select_dtypes(include=['number'])
    
    # 2. Calculate the correlation matrix
    corr_matrix = numerical_df.corr()
    
    # 3. Create the plot
    plt.figure(figsize=(10, 8))
    
    # 4. Draw the heatmap
    # annot=True means show the numbers in the boxes
    # cmap='coolwarm' sets the color scheme (Red=High positive, Blue=High negative)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    
    plt.title('Correlation Heatmap of Flight Variables')
    plt.show()

    # --- Visualization 5: Plot Days Before Departure vs Price ---
def plot_days_vs_price(df):
    """
    Scatter plot to check if booking earlier leads to lower prices.
    """
    plt.figure(figsize=(10, 6))
    
    # Draw the scatter plot
    # alpha=0.3 makes points transparent so you can see where they overlap
    sns.scatterplot(data=df, x='days_before_departure', y='price', alpha=0.3, color='blue')
    
    plt.title('Flight Price vs. Days Before Departure')
    plt.xlabel('Days Before Departure (The larger, the earlier you book)')
    plt.ylabel('Ticket Price (BDT)')
    
    # Add a grid for easier reading
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()

    # --- Visualization 6: Plot Price Distribution by Class ---
def plot_class_price_distribution(df):
    """
    Boxplot to compare the price range of Economy vs Business class.
    """
    plt.figure(figsize=(8, 6))
    
    # Draw the boxplot
    # x='class' divides data by category
    # y='price' shows the value distribution
    # palette='Set2' makes the colors look nice
    sns.boxplot(data=df, x='class', y='price', hue='class', palette='Set2', legend=False)
    
    plt.title('Price Distribution: Economy vs. Business Class')
    plt.xlabel('Flight Class')
    plt.ylabel('Price (BDT)')
    
    # Add grid lines for easier reading
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

    # --- Add these imports at the VERY TOP of your file ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ... (your existing code) ...

# --- Model 1: Train Linear Regression Model ---
def train_linear_regression(df):
    """
    Builds a Multiple Linear Regression model and plots Actual vs. Predicted prices.
    """
    print("--- Starting Linear Regression Training ---")
    
    # 1. Prepare Features (X) and Target (y)
    drop_cols = ['price', 'base_fare_(bdt)', 'tax_&_surcharge_(bdt)', 'dep_time', 'arr_time']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['price']
    
    # 2. Convert text columns to numbers (One-Hot Encoding)
    print("Encoding categorical data...")
    X = pd.get_dummies(X, drop_first=True)
    
    # 3. Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Make predictions
    y_pred = model.predict(X_test)
    
    # 6. Calculate Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Performance:")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.0f}")
    print("-" * 30)
    
    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of Actual vs Predicted
    plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
    
    # Draw a diagonal red line (Perfect Prediction Line)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    
    plt.xlabel('Actual Price (BDT)')
    plt.ylabel('Predicted Price (BDT)')
    plt.title(f'Linear Regression: Actual vs. Predicted Prices\nR2 Score: {r2:.2f}')
    plt.grid(True)
    plt.show()
    
    return model

# --- Data Preparation (Encoding & Splitting) ---
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


# --- Train Regression Models (Linear & Random Forest) ---
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


# --- Train Classification Model (Decision Tree) ---
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

