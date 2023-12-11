import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import json

def feature_process_noreview(train_data):
    # Convert "yelping_since" to "yelping_duration":
    collection_date = pd.to_datetime("2018-07-02") # latest review date
    train_data['yelping_since'] = pd.to_datetime(train_data['yelping_since'])
    train_data['yelping_duration'] = (collection_date - train_data['yelping_since']).dt.days

    # Convert "friends" to "num_of_friends":
    train_data['num_of_friends'] = train_data['friends'].apply(lambda x: 0 if x == "None" else len(x.split(',')))

    # Convert "elite" to "num_of_elites":
    train_data['num_of_elites'] = train_data['elite'].apply(lambda x: 0 if x == "None" else len(x.split(',')))

    train_data.drop(['yelping_since', 'friends', 'elite'], axis=1, inplace=True)
    
    return train_data

def average_category_embeddings(category_string, category_embeddings, embedding_size):
    if category_string == None:
        return [0] * embedding_size
    cleaned_categories = [cat.replace(' ', '_') + '-c' for cat in category_string.split(", ")]
    embeddings = category_embeddings.loc[category_embeddings.index.intersection(cleaned_categories)]
    if embeddings.empty:
        return [0] * embedding_size
    return embeddings.mean().tolist()

# Function to process embeddings file and filter based on IDs
def filter_embeddings(embedding_file, ids_to_keep, id_column_name):
    filtered_embeddings = []
    with open(embedding_file, 'r') as file:
        for line in file:
            entity_id = line.split('\t')[0]
            if entity_id in ids_to_keep:
                filtered_embeddings.append(line.strip().split('\t'))

    # Convert to DataFrame
    df = pd.DataFrame(filtered_embeddings)
    df.set_index(0, inplace=True)  # Set index as entity_id
    df.columns = [f'{id_column_name}_eb{i}' for i in range(1, df.shape[1] + 1)]
    return df

def json2df(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = map(json.loads, f.readlines())
        df = pd.DataFrame.from_records(lines)
    return df

if __name__ == '__main__': 
    
    start_time = time.time()

    input_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    
    # 1. Load the JSON data into a DataFrame
    user_df = json2df(input_file_path + "/user.json")
    business_df = json2df(input_file_path + "/business.json")
    review_df = json2df(input_file_path + "/review_train.json")   
    
    print("Feature loaded------")
    # 2. Load the training data
    #train_data = pd.read_csv(input_file_path + "/yelp_train.csv")
    #val_data = pd.read_csv(input_file_path + "/yelp_val.csv")
    test_data = pd.read_csv(test_file_path)[['user_id', 'business_id']]

    # 3. Merge user and business dataset
    #train_data = pd.merge(train_data, user_df, on="user_id", how="left")
    #train_data = pd.merge(train_data, business_df, on="business_id", how="left")
    #val_data = pd.merge(val_data, user_df, on="user_id", how="left")
    #val_data = pd.merge(val_data, business_df, on="business_id", how="left")   
    test_data = pd.merge(test_data, user_df, on="user_id", how="left")
    test_data = pd.merge(test_data, business_df, on="business_id", how="left")
    
    # Add suffix to "user_id" and "business_id"
    #train_data['user_id'] = train_data ['user_id'].astype(str) + '-u'
    #train_data['business_id'] = train_data['business_id'].astype(str) + '-b'
    test_data['user_id'] = test_data ['user_id'].astype(str) + '-u'
    test_data['business_id'] = test_data['business_id'].astype(str) + '-b'
    
    print("JSON Data merged------")
    end_time = time.time()
    print("Duration: ", end_time - start_time)
    
    print("Start merging with PBG embedding------")
    user_embeddings = pd.read_csv('./node_embeddings/filtered_user_embeddings.csv', index_col=0)
    business_embeddings = pd.read_csv('./node_embeddings/filtered_business_embeddings.csv', index_col=0)
    city_embeddings_file = './node_embeddings/city_embeddings.tsv'
    category_embeddings_file = './node_embeddings/category_embeddings.tsv'
    
    # Merge user and business embeddings with both training and test datasets
    test_data = test_data.merge(user_embeddings, how='left', left_on='user_id', right_index=True)
    test_data = test_data.merge(business_embeddings, how='left', left_on='business_id', right_index=True)
    print("User & Business embeddings merged------")
    end_time = time.time()
    print("Duration: ", end_time - start_time)
    
    # Merge city embeddings with both training and validation datasets
    # Clean "city"
    test_data['city'] = test_data['city'].astype(str) + '-ct'
    test_data['city'] = test_data['city'].str.strip().str.lower().str.replace(' ', '_')

    unique_city_ids = set(test_data['city'])
    city_embeddings = filter_embeddings(city_embeddings_file, unique_city_ids, 'city')

    test_data = test_data.merge(city_embeddings, how='left', left_on='city', right_index=True)
    
    print("City embeddings merged------")
    end_time = time.time()
    print("Duration: ", end_time - start_time)
    
    # Merge category embeddings with both training and validation datasets
    category_embeddings = pd.read_csv(category_embeddings_file, sep='\t', header=None, index_col=0)
    category_embeddings.columns = [f'category_eb{i}' for i in range(1, category_embeddings.shape[1] + 1)]

    # The number of dimensions in your embeddings is 100
    embedding_size = 100
    category_embedding_cols = [f'category_eb{i}' for i in range(1, embedding_size + 1)]

    test_avg_category_embeddings = test_data['categories'].apply(lambda x: average_category_embeddings(x, category_embeddings, embedding_size))

    # Convert lists of embeddings to a DataFrame   
    test_avg_category_embeddings_df = pd.DataFrame(test_avg_category_embeddings.tolist(), index=test_data.index, columns=category_embedding_cols)

    # Concatenate the new DataFrame with the original training and validation data
    test_data = pd.concat([test_data, test_avg_category_embeddings_df], axis=1)

    # Drop the original 'categories' column if no longer needed
    test_data.drop('categories', axis=1, inplace=True)
    
    print("Category embeddings merged------")
    end_time = time.time()
    print("Duration: ", end_time - start_time)
    
    # 4. Drop features
    columns_to_drop = [
    'user_id', 'business_id',   # Identifiers
    'name_x', 'name_y',        # User and business names (assuming name_x is from the user dataset and name_y is from the business dataset)
    'address', 'postal_code',  # Specific location details which might not be generalizable
    'hours',                   # Detailed operational hours (unless you plan to engineer features from it)
    'neighborhood',            # Can be dropped if you're using city and state for location info
    'attributes',              # Might be complex to parse, but can be useful if processed correctly      
    'city', 'state'            # Use Latitude and Longitude
]
    
    #train_data = train_data.drop(columns=columns_to_drop)
    test_data = test_data.drop(columns=columns_to_drop)
    
    #train_data = feature_process_noreview(train_data)
    test_data = feature_process_noreview(test_data)
    
    # Convert all embedding columns to float at once
    test_data.fillna(0, inplace=True)
    embedding_columns = [col for col in test_data.columns if 'eb' in col]
    test_data[embedding_columns] = test_data[embedding_columns].astype(float)
    
    print("Feature processing done!")
    
    print("Number of features: " + str(len(test_data.columns)))
    end_time = time.time()
    print("Duration: ", end_time - start_time)
    
    print("Ready for prediction")
    
    # 5. Load the model and predict
    bst = xgb.Booster()  
    bst.load_model('model-TrainVal-ubcctEmbedding426-epoch100-RMSE890982.json')   

    dtest = xgb.DMatrix(test_data)
    y_pred = bst.predict(dtest)
    
    ids_data = pd.read_csv(test_file_path)[['user_id', 'business_id']]
    
    result = pd.DataFrame({
        'user_id': ids_data['user_id'],
        'business_id': ids_data['business_id'],
        'prediction': y_pred
    })
    
    # Save to CSV
    result.to_csv(output_file_path , index=False)

    end_time = time.time()
    print("Duration: ", end_time - start_time)
    