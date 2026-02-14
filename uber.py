import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Uber Ride Analysis Dashboard", layout="wide")

st.title("Uber Ride Demand & Driver Performance Dashboard")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    st.success("Dataset loaded successfully!")
else:
    st.info("Using synthetic dataset...")
    # Generate synthetic dataset
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "ride_id": range(1, n+1),
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="H"),
        "pickup_location": np.random.choice(['Downtown', 'Airport', 'Suburb', 'Station'], n),
        "dropoff_location": np.random.choice(['Downtown', 'Airport', 'Suburb', 'Station'], n),
        "distance_km": np.round(np.random.uniform(1, 30, n), 2),
        "ride_duration_min": np.round(np.random.uniform(5, 90, n), 2),
        "surge_multiplier": np.round(np.random.choice([1, 1.2, 1.5, 2, 2.5], n), 2),
        "fare_amount": np.round(np.random.uniform(5, 100, n), 2),
        "driver_id": np.random.choice([f'Driver_{i}' for i in range(1, 21)], n),
        "driver_rating": np.round(np.random.uniform(3.5, 5, n), 1),
        "customer_rating": np.round(np.random.uniform(3, 5, n), 1),
        "payment_type": np.random.choice(['Cash', 'Card', 'Digital'], n)
    })

st.dataframe(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
st.subheader("Data Cleaning")
missing_values = df.isnull().sum()
st.write("Missing values per column:", missing_values)

# -----------------------------
# Feature Engineering
# -----------------------------
st.subheader("Feature Engineering")
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['fare_per_km'] = df['fare_amount'] / df['distance_km']

st.dataframe(df.head())

# -----------------------------
# Data Normalization
# -----------------------------
st.subheader("Normalization of Numeric Features")
numeric_cols = ['distance_km', 'ride_duration_min', 'fare_amount', 'fare_per_km', 'driver_rating', 'customer_rating']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
st.dataframe(df.head())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
st.subheader("Exploratory Data Analysis & Visualizations")

# Demand by hour
st.write("### Ride Demand by Hour")
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='hour', palette='viridis')
st.pyplot(plt)

# Surge Analysis
st.write("### Surge Multiplier Distribution")
plt.figure(figsize=(8,4))
sns.histplot(df['surge_multiplier'], bins=10, kde=True, color='orange')
st.pyplot(plt)

# Driver Performance
st.write("### Average Driver Rating")
avg_rating = df.groupby('driver_id')['driver_rating'].mean().sort_values(ascending=False)
st.bar_chart(avg_rating)

# Fare vs Distance
st.write("### Fare vs Distance")
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='distance_km', y='fare_amount', hue='surge_multiplier', palette='coolwarm')
st.pyplot(plt)

# -----------------------------
# Hypothesis Testing Example
# -----------------------------
st.subheader("Hypothesis Testing")
st.write("Do rides with surge > 1.5 have higher fare per km?")
high_surge = df[df['surge_multiplier'] > 1.5]['fare_per_km']
low_surge = df[df['surge_multiplier'] <= 1.5]['fare_per_km']
st.write(f"Average fare per km for high surge: {high_surge.mean():.2f}")
st.write(f"Average fare per km for low surge: {low_surge.mean():.2f}")

# -----------------------------
# Predictive Model Example
# -----------------------------
st.subheader("Predict Fare Amount Prediction")
X = df[['distance_km', 'ride_duration_min', 'surge_multiplier']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

st.write(f"Model Coefficients: {model.coef_}")
st.write(f"Model Intercept: {model.intercept_}")

st.write("Sample Predictions:")
st.dataframe(pd.DataFrame({'Actual': y_test.values, 'Predicted': np.round(pred,2)}).head())
