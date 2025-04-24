import pandas as pd

# Load the dataset from a CSV file
df = pd.read_csv('wine.csv')

# Calculate the statistics for each column
statistics = pd.DataFrame({
    'Mean': df.mean(),
    'Median': df.median(),
    'Mode': df.mode().iloc[0],  # Take the first mode in case there are multiple
    'Standard Deviation': df.std(),
    'Variance': df.var(),
    'Range': df.max() - df.min()
})

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Automatically adjust the width of the display
pd.set_option('display.max_colwidth', None)  # Ensure no truncation of column content

# Print the statistics in a tabular format
print(statistics)