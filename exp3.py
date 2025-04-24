import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('movies.csv')

# 1. Clean 'YEAR' - Remove parentheses and convert to numeric
df['YEAR'] = df['YEAR'].str.extract('(\d{4})').astype(float)  # Extract the year in YYYY format

# 2. Clean 'GENRE' - Remove extra spaces and quotes
df['GENRE'] = df['GENRE'].str.strip().str.replace('"', '').str.replace('            ', ', ')  # Clean genres

# 3. Clean 'RATING' - Handle missing values (fill with the mean rating)
df['RATING'] = pd.to_numeric(df['RATING'], errors='coerce')  # Convert to numeric, coercing errors (NaN)
df['RATING'] = df['RATING'].fillna(df['RATING'].mean())  # Fill missing ratings with the mean

# 4. Clean 'VOTES' - Remove commas and convert to numeric
df['VOTES'] = df['VOTES'].str.replace(',', '').astype(float)  # Remove commas and convert to float
df['VOTES'] = df['VOTES'].fillna(df['VOTES'].mean())  # Fill missing votes with the mean

# 5. Clean 'Gross' - Remove dollar sign, 'M' or 'B' and convert to numeric
def clean_gross(value):
    if isinstance(value, str):
        value = value.replace('$', '').strip()  # Remove the dollar sign
        if 'M' in value:
            value = float(value.replace('M', '').strip()) * 1_000_000  # Convert millions to numeric
        elif 'B' in value:
            value = float(value.replace('B', '').strip()) * 1_000_000_000  # Convert billions to numeric
        else:
            value = float(value)  # If no 'M' or 'B', just convert directly
    return value

df['Gross'] = df['Gross'].apply(clean_gross)  # Apply the function to the 'Gross' column
df['Gross'] = df['Gross'].fillna(df['Gross'].mean())  # Fill missing gross values with the mean

# 6. Clean 'RunTime' - Remove spaces, ensure it's a string first, and then convert to numeric
df['RunTime'] = df['RunTime'].astype(str).str.replace(' ', '').astype(float)  # Ensure it's a string first and remove spaces
df['RunTime'] = df['RunTime'].fillna(df['RunTime'].mean())  # Fill missing runtime with the mean

# 7. Clean 'STARS' - Remove unwanted characters and extra spaces
df['STARS'] = df['STARS'].str.replace(r'\n|\r', ', ').str.replace('Director:|Stars:', '').str.strip()  # Clean stars

# 8. Clean 'ONE-LINE' - Trim extra spaces and clean any unwanted characters
df['ONE-LINE'] = df['ONE-LINE'].str.strip()  # Strip leading/trailing spaces

# 9. Handle missing values for 'GENRE' (optional - fill with mode)
df['GENRE'] = df['GENRE'].fillna('Unknown')  # Fill missing genre with 'Unknown'

# 10. Handle any remaining missing values for non-numeric columns
df['STARS'] = df['STARS'].fillna('Unknown')  # Fill missing stars with 'Unknown'

# For numeric columns, fill missing values with the mean (already done above)
# Final cleaned dataset
print("\nCleaned dataset:\n", df.head())

# Optionally, save the cleaned dataset to a new CSV file
df.to_csv('cleaned_movies.csv', index=False)