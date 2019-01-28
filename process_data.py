import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# load datasets from csv files
messages = pd.read_csv("messages.csv")
categories = pd.read_csv("categories.csv")

# join the two df's
df = messages.join(categories, rsuffix="_categories")

# One-hot-encode the categories
categories = df["categories"].str.split(";", expand=True) # Expand the category column to individual category columns

# Extract the values from the category strings and enter the values into corresponding columns
category_colnames = lambda x: [str(y)[:-2] for y in x]
row = categories.iloc[0]
categories.columns = category_colnames(row)

for column in categories.columns:
    categories[column] = categories[column].apply(lambda x: str(x)[-1])
    categories[column] = categories[column].apply(lambda x: int(x))

# Clean up and add the categories df
df.drop(["id_categories", "categories"], axis=1, inplace=True)
df = pd.concat([df, categories], axis=1)

# Drop duplicates
df.drop_duplicates(keep=False, inplace=True)

# Save the df to the SQL db
engine = create_engine('sqlite:///DisasterResponse.db')
df.to_sql('Messages_Categories', engine, index=False)
