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
data_set = pd.concat([fake_df, true_df])

#Shuffle dataframe
data_set = data_set.sample(frac=1, random_state=42).reset_index(drop=True)

#Send to csv for final dataset observation
#data_set.to_csv("final_preprocessed_ds.csv")

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

###train_naive_bayes function trains the model using naive bayes, returning all necessary probabilities for classification
def train_naive_bayes(train_set, V):
    #prior probabilites
    false_count, true_count = train_set.value_counts(subset=['label'])
    p_false = false_count / len(train_set)
    p_true = true_count / len(train_set)

    #conditional probabilites
    #split df into false instances
    false_df = train_set[train_set['label'] == 'False']
    false = {}; false_count = 0
    #Loop over each instance of false, to create the frequency dictionary
    #Find count of each word in the false label text
    for _, row in false_df.iterrows():
        sent = row['text']
        for i in sent.split():
            false_count += 1
            if i in false:
                false[i] += 1
            else:
                false[i] = 1

    p_word_given_false = {}
    #Loop through false dict and calculate prbabilities for each word given classification of false
    for key, val in false.items():
        prob = (val + 1) / (false_count + V)
        p_word_given_false[key] = prob

    #Follow the same procedure above with true set
    true_df = train_set[train_set['label'] == 'True']
    true = {}; true_count = 0
    for _, row in true_df.iterrows():
        sent = row['text']
        for i in sent.split():
            true_count += 1
            if i in true:
                true[i] += 1
            else:
                true[i] = 1

    p_word_given_true = {}
    for key, val in true.items():
        prob = (val + 1) / (true_count + V)
        p_word_given_true[key] = prob
    
    #Return prior and conditional probabilites
    return p_false, p_true, p_word_given_false, p_word_given_true

###test_naive_bayes tests the model on the test set or the sentence that the user inputs and returns either confusion matrix metrics or sentence probabilites
def test_naive_bayes(test, p_false, p_true, p_word_given_false, p_word_given_true):
    false = []
    true = []
    sentence_false_acc = 0
    sentence_true_acc = 0
    #Check if the input is a dataframe or a sentence
    if isinstance(test, pd.DataFrame):
        #Loop through each sentence of test set
        for _, row in test.iterrows():
            p_false_given_sentence = 1
            p_true_given_sentence = 1
            sent = row['text']
            #Loop through words of sentence
            for word in sent.split():
                #If word is in the probability dictionary, multiply to the accumulator
                if word in p_word_given_false:
                    p_false_given_sentence *= p_word_given_false[word]
                if word in p_word_given_true:
                    p_true_given_sentence *= p_word_given_true[word]
                #If word is not, skip
            #Multiply accumulator to the prior probability
            p_false_given_sentence *= p_false
            p_true_given_sentence *= p_true
            #Append to the list of probabilites of the test set
            false.append(p_false_given_sentence)
            true.append(p_true_given_sentence)
    else:
        #Preprocess, remove non-words, digits, spaces
        preprocess_txt = re.sub(r'[^\w\s]', '', test)
        preprocess_txt = re.sub(r'\d+', '', preprocess_txt)
        preprocess_txt = re.sub(r' +', ' ', preprocess_txt)
        #Loop through words of sentence
        for word in preprocess_txt.split():
            #If word is in the probability dictionary
            if word in p_word_given_false:
                #Take log and 'add' it to accumulator
                sentence_false_acc += math.log(p_word_given_false[word], 2)
            if word in p_word_given_true:
                sentence_true_acc += math.log(p_word_given_true[word], 2)
        #Add accumulator to the log of the prior probability
        sentence_false_acc += math.log(p_false, 2)
        sentence_true_acc += math.log(p_true, 2)
        sentence_false_prob = pow(2, sentence_false_acc)
        sentence_true_prob = pow(2, sentence_true_acc)
    
    index = 0
    metric = { 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0 }
    #If test instance is dataframe:
    if isinstance(test, pd.DataFrame):
        #zip true and false lists and loop
        probs = zip(false, true)
        for false_prob, true_prob in probs:
            #Access value of corresponding label in dataframe
            label = test['label'].iloc[index]
            index += 1
            #If true is greater than false, increment 
            if true_prob > false_prob:
                #If it matches label true in the dataset
                if label == 'True':
                    metric['tp'] += 1
                else:
                    metric['fp'] += 1
            else:
                #If it matches label false
                if label == 'False':
                    metric['tn'] += 1
                else:
                    metric['fn'] += 1
        return metric['tp'], metric['fp'], metric['tn'], metric['fn']
    else:
        return sentence_false_prob, sentence_true_prob

###metric outputs the passed in and derived metric values
def metric(tp, fp, tn, fn):
    print("Number of true positives:", tp)
    print("Number of true negatives:", tn)
    print("Number of false positives:", fp)
    print("Number of false negatives:", fn)
    print("Sensitivity (recall):", tp / (tp + fn))
    print("Specificity:", tn / (tn + fp))
    print("Precision:", tp / (tp + fp))
    print("Negative predictive value:", tn / (tn + fn))
    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))
    print("F1-score:", 2 * ( ( (tp / (tp + fp)) * (tp / (tp + fn)) ) / ( (tp / (tp + fp)) + (tp / (tp + fn)) ) ))


#vocab: count all words in text column without repeting words
V = 0; vocab = set()
for text in data_set['text']:
    vocab.update(set(text.split()))
V = len(vocab)

#Split data
train_set, test_set = train_test_split(data_set, TRAIN_SIZE)

###Console output (where the models are called and run and sentences is classified)
print("Haider, Mazin, A20422384 solution:")
print("Training set size:", TRAIN_SIZE,"%")
if ALGO == 0:
    print("Classifier type: Naive Bayes")
    print("\nTraining classifier...")
    p_false, p_true, p_word_given_false, p_word_given_true = train_naive_bayes(train_set, V)
    print("Testing classifier...")
    tp, fp, tn, fn = test_naive_bayes(test_set, p_false, p_true, p_word_given_false, p_word_given_true)
    print("\nTest results / metrics:")
    metric(tp, fp, tn, fn)

    option = 'Y'
    while (option[0].lower().strip() == 'y'):
        print()
        sentence = input("Enter your sentence/document: ")
        sent_p_false, sent_p_true = test_naive_bayes(sentence, p_false, p_true, p_word_given_false, p_word_given_true)
        print("\nSentence/document S:", sentence)
        if sent_p_true > sent_p_false:
            print("was classified as True")
        else:
            print("was classified as False")
        print("P(False | S) =", sent_p_false)
        print("P(True | S) =", sent_p_true)
        print()
        option = input("Do you want to enter another sentence [Y/N]? ")
else:
    print("Classifier type: k-NN")



