import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

st.title("Used Car Dataset Analysis")

# Load the dataset directly
df = pd.read_csv('used_car_dataset.csv')

st.write("### Raw Dataset:")
st.dataframe(df.head())

# Data Info
st.write("### Dataset Information")
buffer = StringIO()  # Use StringIO as buffer
df.info(buf=buffer)  # Pass the StringIO buffer
st.text(buffer.getvalue())  # Get the string value from the buffer

# Data Cleaning
st.write("### Data Cleaning")
df['kmDriven'] = pd.to_numeric(
    df['kmDriven'].str.replace('km', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip(), errors='coerce')

# KNN Imputation
st.write("Filling NaN values with K-Nearest Neighbor")
features_for_imputation = df[['kmDriven']].copy()
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_for_imputation)
knn_imputer = KNNImputer(n_neighbors=5)
features_imputed = knn_imputer.fit_transform(features_scaled)
features_imputed = scaler.inverse_transform(features_imputed)
df['kmDriven'] = features_imputed[:, 0]

# Date Transformation
df['PostedDate'] = pd.to_datetime(df['PostedDate'], errors='coerce', format='%b-%y')
df['PostedDate'] = df['PostedDate'].apply(lambda x: x.replace(day=1) if pd.notnull(x) else x)

# AskPrice Transformation
df['AskPrice'] = df['AskPrice'].str.replace('₹', '').str.replace(',', '').astype('float')

st.write("### Cleaned Dataset:")
st.dataframe(df.head())

# Exploratory Data Analysis
st.write("### Exploratory Data Analysis")

# Distribution of Car Prices
st.subheader("Distribution of Car Prices")
fig, ax = plt.subplots()
sns.histplot(df['AskPrice'], bins=30, kde=True, ax=ax)
ax.set(title='Distribution of Car Prices', xlabel='Ask Price', ylabel='Frequency')
st.pyplot(fig)

# Mileage vs Price
st.subheader("Mileage vs. Price")
fig, ax = plt.subplots()
sns.scatterplot(x='kmDriven', y='AskPrice', data=df, ax=ax)
ax.set(title='Mileage vs. Price', xlabel='Kilometers Driven', ylabel='Ask Price')
st.pyplot(fig)

# Fuel Type vs Price
st.subheader("Car Price Distribution by Fuel Type")
fig, ax = plt.subplots()
sns.boxplot(x='FuelType', y='AskPrice', data=df, ax=ax)
ax.set(title='Car Price Distribution by Fuel Type', xlabel='Fuel Type', ylabel='Ask Price')
st.pyplot(fig)

# Transmission vs Price
st.subheader("Car Price Distribution by Transmission")
fig, ax = plt.subplots()
sns.boxplot(x='Transmission', y='AskPrice', data=df, ax=ax)
ax.set(title='Car Price Distribution by Transmission', xlabel='Transmission', ylabel='Ask Price')
st.pyplot(fig)

# Owner Type vs Price
st.subheader("Car Price Distribution by Owner Type")
fig, ax = plt.subplots()
sns.boxplot(x='Owner', y='AskPrice', data=df, ax=ax)
ax.set(title='Car Price Distribution by Owner Type', xlabel='Owner Type', ylabel='Ask Price')
st.pyplot(fig)

# Posted Date Analysis
st.subheader("Number of Car Listings Over Time")
fig, ax = plt.subplots()
df['PostedDate'].value_counts().sort_index().plot(kind='line', ax=ax)
ax.set(title='Number of Car Listings Over Time', xlabel='Posted Date', ylabel='Number of Listings')
st.pyplot(fig)

# Correlation Matrix
st.subheader("Correlation Matrix")
correlation_matrix = df.select_dtypes(include=['number']).corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set(title='Correlation Matrix')
st.pyplot(fig)

# Data Visualization
st.write("### Data Visualization")

# Year Distribution of Cars
st.subheader("Buy Year Distribution of Cars")
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(df['Year'], bins=30, kde=True, ax=ax)
ax.set(title='Buy Year Distribution of Cars', xlabel='Year', ylabel='Number of Cars')
st.pyplot(fig)
st.write("This plot shows that most cars in the dataset were bought between 2010 and 2020. There's a gradual increase in the number of cars bought in more recent years, indicating a potential trend towards newer vehicles.")

# Top 10 Car Brands
st.subheader("Top 10 Car Brands")
car_brand_counts = df['Brand'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=car_brand_counts.index, y=car_brand_counts.values, palette='viridis', ax=ax)
ax.set(title='Top 10 Car Brands', xlabel='Car Brand', ylabel='Count')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
st.write("The 'Top 10 Car Brands' plot reveals that Maruti is the most frequent car brand in the dataset, followed by Hyundai and Mahindra. These brands have significantly higher counts compared to other brands, suggesting their popularity in the used car market. Brands like Tata, Ford, and Honda have moderate counts, while Renault, Volkswagen, and Chevrolet appear less frequently.")


# Average Ask Price by Transmission Type
st.subheader("Average Ask Price by Transmission Type")
average_ask_price_by_transmission = df.groupby('Transmission')['AskPrice'].mean()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=average_ask_price_by_transmission.index, y=average_ask_price_by_transmission.values, palette='viridis', ax=ax)
ax.set(title='Average Ask Price by Transmission Type', xlabel='Transmission', ylabel='Average Ask Price')
st.pyplot(fig)
st.write("The 'Average Ask Price by Transmission Type' plot shows that cars with automatic transmissions have a considerably higher average asking price compared to cars with manual transmissions. This suggests that automatic transmission is a desirable feature in the used car market, potentially commanding a price premium.")

# Asking Price of Different Transmissions for Top 10 Car Brands
st.subheader("Total Asking Price of Different Transmissions for Top 10 Car Brands")
top_10_brands = df['Brand'].value_counts().nlargest(10).index
df_top_10 = df[df['Brand'].isin(top_10_brands)]
brand_transmission_price = df_top_10.groupby(['Brand', 'Transmission'])['AskPrice'].mean().reset_index()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x='Brand', y='AskPrice', hue='Transmission', data=brand_transmission_price, palette='viridis', ax=ax)
ax.set(title='Total Asking Price of Different Transmissions for Top 10 Car Brands', xlabel='Car Brand', ylabel='Average Asking Price')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
st.write("The 'Total Asking Price of Different Transmissions for Top 10 Car Brands' plot shows that, for most of the top 10 brands, cars with automatic transmissions have higher average asking prices compared to their manual counterparts. This indicates that transmission preference can vary depending on the brand and specific car model.")

# Average Price per Brand
st.subheader("Average Price per Brand (Sorted)")
average_price_per_brand = df.groupby('Brand')['AskPrice'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=average_price_per_brand.index, y=average_price_per_brand.values, palette='viridis', ax=ax)
ax.set(title='Average Price per Brand (Sorted)', xlabel='Brand', ylabel='Average Price')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
st.write("The 'Average Price per Brand (Sorted)' plot reveals that brands like Aston Martin, Rolls-Royce, and Bentley have the highest average asking prices, indicating their premium positioning in the used car market. Brands like Maruti, Hyundai, and Mahindra have relatively lower average prices, suggesting their affordability and popularity among budget-conscious buyers. Other brands fall in between, representing a range of price points and market segments.")