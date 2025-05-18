import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
import pickle
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def one_hot_encoding(df):
    df['cuisine'] = df['cuisine'].str.lower().str.strip()
    df['cuisine_list'] = df['cuisine'].str.split(',').apply(lambda x: [i.strip() for i in x])

    # Label Encoders
    le_city = LabelEncoder()
    le_name = LabelEncoder()

    # Encode city and name (as Series)
    df['city_main_encoded'] = le_city.fit_transform(df['city_main'])
    df['name_encoded'] = le_name.fit_transform(df['name'])

    # Cuisine (MultiLabelBinarizer)
    mlb = MultiLabelBinarizer()
    cuisine_df = pd.DataFrame(mlb.fit_transform(df['cuisine_list']), columns=mlb.classes_)
    df_encoded = pd.concat([df.reset_index(drop=True), cuisine_df.reset_index(drop=True)], axis=1)
    

    # Save the encoders
    os.makedirs('data', exist_ok=True)
    with open('data/mlb_cuisine_encoder.pkl', 'wb') as file:
        pickle.dump(mlb, file)

    with open('data/label_encoder_city.pkl', 'wb') as file:
        pickle.dump(le_city, file)

    with open('data/label_encoder_name.pkl', 'wb') as file:
        pickle.dump(le_name, file)

    return df_encoded

if __name__ == '__main__':
    file_path = "D:/Data Science Projects/Swiggy Recommendation System/data/cleaned_data_after_oh.csv"
    loaddata = load_data(file_path)

    df_encoded = one_hot_encoding(loaddata)
    df_encoded.to_csv('data/encoded_data.csv', index=False)
    print("✅ Encoding complete! Encoded data saved to 'data/encoded_data.csv'")
