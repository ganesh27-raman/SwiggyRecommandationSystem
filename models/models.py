import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def cuisine_match(row_cuisines, selected_cuisines):
    row_set = set(map(str.strip, row_cuisines.lower().split(',')))
    selected_set = set(c.lower().strip() for c in selected_cuisines)
    return not row_set.isdisjoint(selected_set)

def recommend_with_kmeans(user_input, encoded_df, original_df, top_n=5):
    city = user_input['city']
    cuisines = user_input['cuisine']
    min_rating = user_input['rating']
    min_rating_count = user_input['rating_count']

    # Filter original_df by user input
    filtered_df = original_df[
        (original_df['city_main'].str.lower() == city.lower()) &
        (original_df['rating'] >= min_rating) &
        (original_df['rating_count'] >= min_rating_count)
    ]
    filtered_df = filtered_df[filtered_df['cuisine'].apply(lambda x: cuisine_match(x, cuisines))]

    if filtered_df.empty:
        return pd.DataFrame(columns=original_df.columns)

    filtered_encoded = encoded_df.loc[filtered_df.index].select_dtypes(include=['number']).fillna(0)

    n_clusters = min(5, len(filtered_encoded))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(filtered_encoded)
    filtered_df = filtered_df.copy()
    filtered_df['cluster'] = clusters

    target_cluster = filtered_df.iloc[0]['cluster']
    cluster_members = filtered_df[filtered_df['cluster'] == target_cluster]
    cluster_members = cluster_members.drop(filtered_df.index[0], errors='ignore')

    return cluster_members.head(top_n)[original_df.columns]


def recommend_with_cosine(user_input, encoded_df, original_df, top_n=5):
    city = user_input['city']
    cuisines = user_input['cuisine']
    min_rating = user_input['rating']
    min_rating_count = user_input['rating_count']

    filtered_df = original_df[
        (original_df['city_main'].str.lower() == city.lower()) &
        (original_df['rating'] >= min_rating) &
        (original_df['rating_count'] >= min_rating_count)
    ]
    filtered_df = filtered_df[filtered_df['cuisine'].apply(lambda x: cuisine_match(x, cuisines))]

    if filtered_df.empty:
        return pd.DataFrame(columns=original_df.columns)

    filtered_encoded = encoded_df.loc[filtered_df.index].select_dtypes(include=['number']).fillna(0)
    filtered_encoded_norm = normalize(filtered_encoded)

    query_index = filtered_df.index[0]
    query_vector = encoded_df.loc[[query_index]].select_dtypes(include=['number']).fillna(0)
    query_vector_norm = normalize(query_vector)

    similarities = cosine_similarity(query_vector_norm, filtered_encoded_norm)[0]

    similarity_series = pd.Series(similarities, index=filtered_df.index)
    similarity_series = similarity_series.drop(query_index)
    top_indices = similarity_series.nlargest(top_n).index

    return original_df.loc[top_indices]
