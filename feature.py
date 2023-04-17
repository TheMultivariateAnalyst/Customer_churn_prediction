# Import libraries
import pandas as pd

# Load cleaned data
df = pd.read_csv('OnlineRetail_cleaned.csv')

# Create new features
df['PurchaseFrequency'] = df['InvoiceNo'] / (df['Year'].max() - df['Year'] + 1)  # Calculate purchase frequency
df['Recency'] = (df['Year'].max() - df['Year']) * 12 + (df['Month'].max() - df['Month'])  # Calculate recency in months
df['MonetaryValue'] = df['TotalSpending'] / df['Quantity']  # Calculate average monetary value per purchase

# Create target variable
df['Churn'] = (df['Recency'] >= 3) & (df['Recency'] <= 6)  # Create binary variable for customer churn

# Save new dataset
df.to_csv('OnlineRetail_processed.csv', index=False)
