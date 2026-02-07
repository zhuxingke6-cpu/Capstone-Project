import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report


# --- DATA EXTRACTION ---
def load_flight_data(url):
    """Loads dataset from URL with error handling."""
    try:
        df = pd.read_csv(url)
        print(f"✅ Success: Data loaded! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading from URL: {e}")
        return None


# --- DATA CLEANUP ---
def clean_column_names(df):
    """Standardizes column names and renames key metrics."""
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


def process_date_features(df):
    """Extracts month, hour, and day of week from dep_time."""
    df['dep_time'] = pd.to_datetime(df['dep_time'], errors='coerce')
    df = df.dropna(subset=['dep_time'])
    df['dep_month'] = df['dep_time'].dt.month_name()
    df['dep_hour'] = df['dep_time'].dt.hour
    df['day_of_week'] = df['dep_time'].dt.dayofweek
    return df


# --- VISUALIZATIONS ---
def plot_route_price_analysis(df):
    """Visualization 1: Top 10 Routes by Average Price."""
    df_route = df.copy()
    df_route['source'] = df_route['source'].fillna('Unknown').astype(str).str.strip()
    df_route['destination'] = df_route['destination'].fillna('Unknown').astype(str).str.strip()
    df_route['route'] = df_route['source'] + " to " + df_route['destination']

    route_stats = df_route.groupby('route')['price'].mean().sort_values(ascending=False).head(10).reset_index()

    plt.figure(figsize=(12, 8))
    sns.barplot(x='price', y='route', data=route_stats, palette='magma')
    plt.title('Visualization 1: Top 10 Most Expensive Flight Routes', fontsize=16)
    plt.xlabel('Average Ticket Price (BDT)')
    plt.ylabel('Flight Route')
    plt.tight_layout()
    plt.show()


def plot_price_distribution_boxplot(df):
    """Visualization 2: Flight Price Distribution by Airline."""
    plt.figure(figsize=(15, 7))
    sns.boxplot(x='airline', y='price', data=df, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Visualization 2: Flight Price Distribution by Airline', fontsize=16)
    plt.show()


def plot_seasonal_price_analysis(df):
    """Visualization 3: Monthly Flight Price Distribution (Seasonality)."""
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='dep_month', y='price', data=df, order=month_order, palette='Set3')
    plt.title('Visualization 3: Monthly Flight Price Distribution', fontsize=16)
    plt.show()


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


def plot_days_vs_price(df):
    """Visualization 5: Days Before Departure vs Price."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='days_before_departure', y='price', alpha=0.3, color='blue')
    plt.title('Visualization 5: Flight Price vs. Days Before Departure')
    plt.show()


def plot_class_price_distribution(df):
    """Visualization 6: Price Distribution by Class."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='class', y='price', palette='Set2')
    plt.title('Visualization 6: Price Distribution by Class')
    plt.show()

# --- Add these imports at the VERY TOP of your file ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ... (your existing code) ...


# --- MODELING PIPELINES ---
def prepare_data(df, target_col, task='regression'):
    """Prepares features and labels for modeling."""
    df_mod = df.copy()
    for col in df_mod.select_dtypes(include=['object']).columns:
        if col != target_col: df_mod[col] = pd.factorize(df_mod[col])[0]
    X = df_mod.drop(columns=[target_col, 'dep_time', 'dep_month'], errors='ignore')
    y = df_mod[target_col]
    if task == 'classification': y = pd.factorize(y)[0]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear_regression(df):
    """Model 1: Multiple Linear Regression."""
    X_train, X_test, y_train, y_test = prepare_data(df, 'price', 'regression')
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Linear Regression R2: {r2_score(y_test, y_pred):.4f}")
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Model 1: Linear Regression - Actual vs Predicted')
    plt.show()
    return model


def run_regression_pipeline(df):
    """Model 2: Random Forest Regressor."""
    X_train, X_test, y_train, y_test = prepare_data(df, 'price', 'regression')
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(f"Random Forest R2: {r2_score(y_test, y_pred):.4f}")
    plt.scatter(y_test, y_pred, alpha=0.3, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Model 2: Random Forest - Actual vs Predicted')
    plt.show()
    return rf


def run_classification_pipeline(df):
    """Model 3: Decision Tree Classifier."""
    X_train, X_test, y_train, y_test = prepare_data(df, 'class', 'classification')
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=list(X_train.columns), filled=True, fontsize=10)
    plt.title('Model 3: Decision Tree Structure')
    plt.show()
    return dt

