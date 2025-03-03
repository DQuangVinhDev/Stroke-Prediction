# Libraries
import pandas as pd 

from sklearn.model_selection import train_test_split

# Read original csv file
df = pd.read_csv (r'Preprocessing\healthcare-dataset-stroke-data.csv', header =0)

# Delete "id" column
data1 = df.drop(columns=['id'], inplace=True)

# Delete "stroke" column at X

X = df.drop("stroke", axis=1)

# Y contains "stroke" column
y = df["stroke"]

# Split data into 2 parts: 80% train data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

### Print length of dataframe, data train and data test
# print("\n")
# print(" Length of data frame: " , len(df),"\n","Length of data train: ",len(X_train),"\n","Length of data test: ", len(X_test))
# print("\n")

# Put X_train and X_test into an array data_splits
data_splits = [X_train,X_test]

# Function "median_filler" used to fill in missing values
def median_filler(split): 

    split["bmi"].fillna(split["bmi"].median(),inplace=True)

# Read array data_splits
for split in data_splits:
    # Run the function "median_filler" and enter the median value in the missing values
    median_filler(split)

    ### Check for missing values
    print(f"Number of nulls: {split['bmi'].isnull().sum()}")
    # print(f"Datatypes:\n{split.dtypes}")
    # print("__________________________________")

# Export to CSV and Excel files
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_excel (r'Preprocessing\DataTrain.xlsx', index = None, header=True)
train_data.to_csv (r'Preprocessing\DataTrain.csv', index = None, header=True) 

test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_excel (r'Preprocessing\DataTest.xlsx', index = None, header=True)
test_data.to_csv (r'Preprocessing\DataTest.csv', index = None, header=True) 

df.to_excel (r'Preprocessing\DataFrame.xlsx', index = None, header=True)

print(" -------- EXPORT COMPLETE -------- ")



