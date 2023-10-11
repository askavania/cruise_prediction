import pandas as pd
from sklearn.model_selection import train_test_split

def convert_mixed_date_format(date_str):
    """
    Convert mixed date formats to a consistent datetime format.
    """
    try:
        # First, attempt to convert 'yyyy-mm-dd' format
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        # If there's an error, try converting 'dd/mm/yyyy' format
        return pd.to_datetime(date_str, format='%d/%m/%Y')

def preprocess_data(pre_df, post_df):
    """
    Preprocess the pre-purchase and post-trip data.
    
    Args:
    - pre_df (DataFrame): Pre-purchase data.
    - post_df (DataFrame): Post-trip data.
    
    Returns:
    - X_train, X_test, y_train, y_test: Train-test split data.
    """
    
    # Dropping Index Column since it is not a relevant feature
    pre_df.drop('index', axis=1, inplace=True)
    post_df.drop('index', axis=1, inplace=True)

    # Convert 'Date of Birth' column to consistent datetime format
    pre_df['Date of Birth'] = pre_df['Date of Birth'].apply(convert_mixed_date_format)

    # Convert 'Logging' column to datetime format (if it's not already)
    pre_df['Logging'] = pd.to_datetime(pre_df['Logging'],format="%d/%m/%Y %H:%M")

    # Calculate age based on the logging year and assign it to a new 'Age' column
    pre_df['Age'] = pre_df['Logging'].dt.year - pre_df['Date of Birth'].dt.year

    # Replace the age of more than 120 with null
    pre_df.loc[pre_df['Age'] > 120, 'Age'] = None

    # Mapping dictionary based on the Importance scale reference
    importance_scale_mapping = {
        'Not at all important': 1.0,
        'A little important': 2.0,
        'Somewhat important': 3.0,
        'Very important': 4.0,
        'Extremely important': 5.0
    }

    # Convert 'Onboard Wifi Service', 'Onboard Entertainment' and 'Onboard Dining Service' columns
    pre_df['Onboard Wifi Service'] = pre_df['Onboard Wifi Service'].map(importance_scale_mapping)
    pre_df['Onboard Entertainment'] = pre_df['Onboard Entertainment'].map(importance_scale_mapping)
    pre_df['Onboard Dining Service'] = pre_df['Onboard Dining Service'].map(importance_scale_mapping)

    # Merging pre_df and post_df on 'Ext_Intcode'
    df = pre_df.merge(post_df, on='Ext_Intcode', how='inner')

    # Categorize age into groups
    bins = [0, 18, 30, 45, 60, 80, 100]
    labels = ['<18', '18-30', '31-45', '46-60', '61-80', '80+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    # Cleaning and converting the 'cruise distance' column to numeric values

    # Extract numeric values and units from the 'cruise distance' column
    df['distance_value'] = df['Cruise Distance'].str.extract('(\d+\.?\d*)').astype(float)
    df['unit'] = df['Cruise Distance'].str.extract('([a-zA-Z]+)')

    # Convert distances from miles to kilometers where unit is 'Miles'. 1 Mile is approximately 1.60934 KM.
    mask_miles = df['unit'] == 'Miles'
    df.loc[mask_miles, 'distance_value'] = df.loc[mask_miles, 'distance_value'] * 1.60934

    # Drop the original 'cruise distance' column and the 'unit' column
    df.drop(['Cruise Distance', 'unit'], axis=1, inplace=True)

    # Rename the 'distance_value' column back to 'cruise distance' and ensure it's of type float
    df.rename(columns={'distance_value': 'Cruise Distance'}, inplace=True)
    df['Cruise Distance'] = df['Cruise Distance'].astype(float)

    # Removing specified columns
    columns_to_remove = ['Logging', 'Ext_Intcode', 'Date of Birth', 'Cruise Name', 'Age Group']
    df.drop(columns_to_remove, axis=1, inplace=True)

    # Dropping rows where our target variable 'Ticket Type' has missing values
    df.dropna(subset=['Ticket Type'], inplace=True)

    # Train Test Split Data
    X = df.drop('Ticket Type', axis=1)
    y = df['Ticket Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Ticket Type'])

    return X_train, X_test, y_train, y_test

