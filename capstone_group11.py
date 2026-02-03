import pandas as pd
import os

# --- Function 1: Load Data ---
def load_flight_data(folder_path):
    """
    Read the CSV file from the downloaded folder.
    读取下载文件夹中的 CSV 文件。
    """
    # Find all filenames in the folder
    # 找到文件夹里的所有文件名
    files = os.listdir(folder_path)
    csv_file = ""
    
    # Look for the file ending with .csv
    # 寻找以 .csv 结尾的文件
    for file in files:
        if file.endswith('.csv'):
            csv_file = file
            break
            
    if csv_file:
        full_path = os.path.join(folder_path, csv_file)
        print(f"Loading file: {csv_file}")
        df = pd.read_csv(full_path)
        return df
    else:
        print("Error: No CSV file found in the directory.")
        return None

# --- Function 2: Clean Column Names ---
def clean_column_names(df):
    """
    Standardize column names to lowercase English for easier coding.
    把复杂的列名改成简单的英文小写，方便后续写代码。
    """
    # 1. Remove whitespace from the beginning and end of column names
    # 1. 去掉列名两边的空格
    df.columns = df.columns.str.strip()
    
    # 2. Define a dictionary mapping old names to new names
    # 2. 定义一个字典，左边是旧名字，右边是新名字
    rename_map = {
        'Departure Date & Time': 'dep_time',
        'Arrival Date & Time': 'arr_time',
        'Total Fare (BDT)': 'price',
        'Day': 'journey_day'
    }
    
    # 3. Execute the renaming process
    # 3. 执行重命名
    df = df.rename(columns=rename_map)
    
    # 4. Convert all column names to lowercase and replace spaces with underscores
    # 4. 把所有列名变成小写，并把空格变成下划线
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    print("Column cleaning complete! Current columns:", list(df.columns))
    return df

# --- Function 3: Process Date Features ---
def process_date_features(df):
    """
    Convert date strings into datetime objects and extract month/hour features.
    将时间字符串转换成 Python 可以理解的时间格式，并提取月份和小时特征。
    """
    # Ensure 'dep_time' is in datetime format
    # 确保 dep_time 是时间格式
    df['dep_time'] = pd.to_datetime(df['dep_time'])
    
    # Extract the month
    # 提取月份
    df['dep_month'] = df['dep_time'].dt.month
    
    # Extract the hour (useful to see if flight time affects price)
    # 提取小时（用来分析是不是半夜的飞机便宜）
    df['dep_hour'] = df['dep_time'].dt.hour
    
    print("Date feature processing complete! Added 'dep_month' and 'dep_hour'.")
    return df
