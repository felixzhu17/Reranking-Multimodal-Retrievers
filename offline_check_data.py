import pandas as pd

# File paths
train_file_path = '/Users/felixzhu/Documents/GitHub/Retrieval-Augmented-Visual-Question-Answering/offline_data/OKVQA_Train_Full_Data.csv'
test_file_path = '/Users/felixzhu/Documents/GitHub/Retrieval-Augmented-Visual-Question-Answering/offline_data/OKVQA_Test_Full_Data.csv'

# Read the CSV files
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Concatenate the DataFrames
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Search for the specific question
search_string = "photo from the"
matching_rows = combined_df[combined_df['question'].str.contains(search_string, na=False)]

# Fire Truck
# "['13716189' '13716188' '1301427' '3398931' '10352771' '6326633' '17895782'\n '10401453']"

# Photo
# "['9624709' '9823882' '16147146' '3374296' '16147150' '3374297' '16147148'\n '16670471' '9624711']"


passage_ids_to_filter = ['9624709', '9823882', '16147146', '3374296', '16147150', '3374297', '16147148', '16670471', '9624711']

# Filter the dataset
filtered_passages = [record for record in test_passages if record['passage_id'] in passage_ids_to_filter]
