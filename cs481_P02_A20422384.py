# Names: Hamza Syed, Mazin Haider

import pandas as pd

#Read in both datasets as pandas df
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

###Preprocessing
#Add fake and true columns for each
fake_df['authenticity'] = 'Fake'
true_df['authenticity'] = 'True'

#Drop duplicate rows
fake_df = fake_df.drop_duplicates()
true_df = true_df.drop_duplicates()

#Merge datasets
df = pd.concat([fake_df, true_df])

#Randomly shuffle datasets between true and false instances
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.head)

