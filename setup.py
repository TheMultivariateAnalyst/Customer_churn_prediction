# Import libraries
import pandas as pd

# Load data
df = pd.read_excel('Online Retail.xlsx')

# Clean and preprocess data
df.dropna(inplace=True)  # Remove rows with missing values
df = df[df['Quantity'] > 0]  # Filter out negative quantity values

# Convert InvoiceDate to datetime format and extract relevant features
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day

# Aggregate data to the customer level
df_customer = df.groupby(['CustomerID'], as_index=False).agg({
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'Year': 'max',
    'Month': 'max',
    'Day': 'max'
})

# Calculate total spending for each customer
df_customer['TotalSpending'] = df['Quantity'] * df['UnitPrice']

# Save cleaned and preprocessed data to a new file
df_customer.to_csv('OnlineRetail_cleaned.csv', index=False)
