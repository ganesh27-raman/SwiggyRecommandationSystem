import streamlit as st
import pandas as pd
from models import recommend_with_kmeans, recommend_with_cosine

# Load data once at startup
original_df = pd.read_csv('D:/Data Science Projects/Swiggy Recommendation System/data/cleaned_data.csv')
encoded_df = pd.read_csv('D:/Data Science Projects/Swiggy Recommendation System/src/data/encoded_data.csv')

st.title("🍽️ Restaurant Recommendation System")

city = st.selectbox("Select City", sorted(original_df['city_main'].unique()))
cuisine = st.multiselect("Select Cuisine(s)", sorted(set(c.strip() for sublist in original_df['cuisine'].str.split(',') for c in sublist)))
rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0)
rating_count = st.slider("Minimum Rating Count", 0, 1000, 20)
method = st.radio("Select Recommendation Method:", ["KMeans Clustering", "Cosine Similarity"])

if st.button("Recommend"):
    user_input = {
        "city": city,
        "cuisine": cuisine,
        "rating": rating,
        "rating_count": rating_count
    }

    if method == "KMeans Clustering":
        results = recommend_with_kmeans(user_input, encoded_df, original_df)
    else:
        results = recommend_with_cosine(user_input, encoded_df, original_df)

    if results.empty:
        st.warning("No matching restaurants found. Try different filters.")
    else:
        st.subheader("🍴 Top Recommendations:")
        st.dataframe(results[['name', 'city_main', 'rating', 'rating_count', 'cuisine']])
