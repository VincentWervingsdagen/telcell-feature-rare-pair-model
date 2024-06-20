import pandas as pd
from sklearn.model_selection import train_test_split

name_of_file = 'common_work'

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('Vincent/{}/output_cell.csv'.format(name_of_file))

# Step 2: Get the unique agents
agents = df['owner'].unique()

# Step 3: Split the agents into training and test sets
train_agents, test_agents = train_test_split(agents, test_size=0.2, random_state=40)

# Step 4: Create training and test DataFrames
train_df = df[df['owner'].isin(train_agents)]
test_df = df[df['owner'].isin(test_agents)]

# Step 5: Save the split DataFrames into separate CSV files
train_df.to_csv('Vincent/{}/training_set_{}.csv'.format(name_of_file,name_of_file), index=False)
test_df.to_csv('Vincent/{}/test_set_{}.csv'.format(name_of_file,name_of_file), index=False)

print("Training and test sets created successfully.")