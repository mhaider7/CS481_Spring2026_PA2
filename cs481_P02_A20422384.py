# Names: Hamza Syed, Mazin Haider

import pandas as pd
import sys, re
from bs4 import BeautifulSoup
import math

#Read in both datasets as pandas df
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

###Preprocessing
#DATE feature
#Check for dates
#for i in fake_df['date'].unique(): print(i, end=" | ")
#for i in true_df['date'].unique(): print(i, end=" | ")
#Drop date column
fake_df.drop(columns=['date'], inplace=True)
true_df.drop(columns=['date'], inplace=True)

#SUBJECT feature
#Check for domain of values
#print(fake_df['subject'].unique())
#print(true_df['subject'].unique())
#Output a csv with only headline and subject to manually observe pattern
#fake_df.to_csv("fake_headline_subject.csv", columns=['title', 'subject'])
#true_df.to_csv("true_headline_subject.csv", columns=['title', 'subject'])
#Change 'Government News' to 'politics', 'US_News' to 'News' 'Middle-East' to 'worldnews' in fake_df
fake_df.loc[fake_df['subject'] == 'Government News', 'subject'] = 'politics'
fake_df.loc[fake_df['subject'] == 'US_News', 'subject'] = 'News'
fake_df.loc[fake_df['subject'] == 'Middle-east', 'subject'] = 'worldnews'
#Change 'politicsNews' to 'politics' in true_df
true_df.loc[true_df['subject'] == 'politicsNews', 'subject'] = 'politics'
#Output dataframe to check if all changes went through for subject column
#fake_df.to_csv("fake_headline_subject_update.csv", columns=['title', 'subject'])
#true_df.to_csv("true_headline_subject_update.csv", columns=['title', 'subject'])

#Drop duplicate rows
fake_df = fake_df.drop_duplicates()
true_df = true_df.drop_duplicates()

#Combine each instance of the dataframes into a single list
FAKE = fake_df.values.tolist()
TRUE = true_df.values.tolist()
#Break down list elements (title, text, subject) into one long string
for i in range(len(FAKE)): FAKE[i] = ' '.join(FAKE[i])
for i in range(len(TRUE)): TRUE[i] = ' '.join(TRUE[i])

#Drop dates in text
for i in range(len(FAKE)): 
    FAKE[i] = re.sub(r'\d+', '', FAKE[i])
    FAKE[i] = re.sub(r'January|February|March|April|May|June|July|August|September|October|November|December', '', FAKE[i])
for i in range(len(TRUE)): 
    TRUE[i] = re.sub(r'\d+', '', TRUE[i])
    TRUE[i] = re.sub(r'January|February|March|April|May|June|July|August|September|October|November|December', '', TRUE[i])

#Drop html tags (Potentially keep if it makes the prediction better)
for i in range(len(FAKE)): FAKE[i] = BeautifulSoup(FAKE[i], "html.parser").get_text()
for i in range(len(TRUE)): TRUE[i] = BeautifulSoup(TRUE[i], "html.parser").get_text()

#Drop social media handlers
for i in range(len(FAKE)): FAKE[i] = re.sub(r'@[^\s]+', '', FAKE[i])
for i in range(len(TRUE)): TRUE[i] = re.sub(r'@[^\s]+', '', TRUE[i])

#note: Pattern matching here takes a long time to compute
#Drop punctuation and extra spaces
for i in range(len(FAKE)): FAKE[i] = re.sub(r'[^\w\s]', '', FAKE[i]); FAKE[i] = re.sub(r' +', ' ', FAKE[i]) 
for i in range(len(TRUE)): TRUE[i] = re.sub(r'[^\w\s]', '', TRUE[i]); TRUE[i] = re.sub(r' +', ' ', TRUE[i])
#for i in range(len(FAKE)): FAKE[i] = re.sub(r'"|!|\.|\?|\(|\)|\[|\]|,|\\|\/|;|:', '', FAKE[i])
#for i in range(len(TRUE)): TRUE[i] = re.sub(r'"|!|\.|\?|\(|\)|\[|\]|,|\\|\/|;|:', '', TRUE[i])

#Lower-case everything
for i in range(len(FAKE)): FAKE[i] = FAKE[i].lower()
for i in range(len(TRUE)): TRUE[i] = TRUE[i].lower()

#Convert to dataframe
fake_df = pd.DataFrame({'text': FAKE, 'label': 'False'})
true_df = pd.DataFrame({'text': TRUE, 'label': 'True'})

#Concat datasets
final_ds = pd.concat([fake_df, true_df])

#Shuffle dataframe
final_ds = final_ds.sample(frac=1, random_state=42).reset_index(drop=True)

#Send to csv for final dataset observation
#final_ds.to_csv("final_preprocessed_ds.csv")

###Command line arguments
# Algo : 0 = Naive Bayes | 1 = k-NN
if len(sys.argv) != 3 or int(sys.argv[2]) < 50 or int(sys.argv[2]) > 90:
    ALGO = 0
    TRAIN_SIZE = 80
else:
    if int(sys.argv[1]) < 0 or int(sys.argv[1]) > 1:
        ALGO = 0
        TRAIN_SIZE = int(sys.argv[2])
    else:
        ALGO = int(sys.argv[1])
        TRAIN_SIZE = int(sys.argv[2])

###train_test_split splits the dataframe based on the size given as an argument
def train_test_split(data_set, train_size):
    train_length = math.floor(len(data_set) * (train_size / 100))
    test_length = len(data_set) - train_length
    train_set = data_set.head(train_length)
    test_set = data_set.tail(test_length)
    return train_set, test_set

