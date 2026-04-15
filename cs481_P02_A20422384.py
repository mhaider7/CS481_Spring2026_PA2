# Names: Hamza Syed, Mazin Haider

import pandas as pd
import sys, re
from bs4 import BeautifulSoup
import math
import heapq
from collections import Counter
from nltk.corpus import stopwords
import nltk

#nltk.download('stopwords')

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

#Load stop words
#Reduces performance, commented out
#stop_words = list(stopwords.words('english'))
#for i in range(len(FAKE)): 
#    fake_words = [word for word in FAKE[i].split() if word not in stop_words]
#    FAKE[i] = ' '.join(fake_words)

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
        
### Helper function to create bag-of-words vector for a document
def create_bow_vector(text, vocab):
    """Create non-binary bag-of-words vector for a document"""
    vector = {}
    words = text.split()
    for word in words:
        if word in vocab:
            vector[word] = vector.get(word, 0) + 1
    return vector

### Function to compute euclidean between two vectors
def euclidean_distance(vec1, vec2):
    """Compute euclidean distance between tf-idf & bag-of-words vectors"""
    # Find common words
    #common_words = set(vec1.keys()) & set(vec2.keys())
    all_words = set(vec1.keys)) | set(vec2.keys())
    #Calculate euclidean dist
    distance = 0
    for word in all_words:
        val1 = vec1.get(word, 0)
        val2 = vec2.get(word, 0)
        distance += math.pow(val1 - val2, 2)
    #distance = math.sqrt(sum((math.pow(vec1[word] - vec2[word], 2)) for word in common_words))
    return math.sqrt(distance)
    #return distance

### Function to train kNN (just store training data)
def train_knn(train_set, vocab):
    """Train kNN by storing training documents and their vectors"""
    train_data = []
    for _, row in train_set.iterrows():
        vector = create_bow_vector(row['text'], vocab)
        train_data.append({
            'vector': vector,
            'label': row['label'],
            'text': row['text']
        })
    
    #Create a frequency dictionary for the number of documents with word for tf-idf computation
    train_data_freq = {}
    for row in train_data:
        for word, _ in row['vector'].items():
            train_data_freq[word] = train_data_freq.get(word, 0) + 1

    return train_data, train_data_freq

def create_tf_idf_vector(instance_freq, num_of_docs, corpus_count):
    tf_idf = {}
    for _, (word, count) in enumerate(instance_freq.items()):
        if word in corpus_count:
            val = count * math.log(num_of_docs/corpus_count[word], 2)
            #Drop any negative words, since they are too frequent and likely stop words
            if val >= 1:
                tf_idf[word] = val
    return tf_idf

### Function to predict using kNN
def predict_knn(test_instance, train_data, k, vocab, num_of_docs, corpus_count):
    """Predict class for a test instance using kNN"""
    # Create bag-of-words vector for test instance
    if isinstance(test_instance, pd.Series):
        # If it's a dataframe row
        test_vector = create_bow_vector(test_instance['text'], vocab)
        test_vector = create_tf_idf_vector(test_vector, num_of_docs, corpus_count)
    else:
        # If it's a string sentence
        # Preprocess the sentence
        preprocess_txt = re.sub(r'[^\w\s]', '', test_instance)
        preprocess_txt = re.sub(r'\d+', '', preprocess_txt)
        preprocess_txt = re.sub(r' +', ' ', preprocess_txt)
        test_vector = create_bow_vector(preprocess_txt.lower(), vocab)
        test_vector = create_tf_idf_vector(test_vector, num_of_docs, corpus_count)

    nearest_neighbors = []
    
    for i, doc in enumerate(train_data):
        dist = euclidean_distance(test_vector, doc['vector'])
        
        # Push to heap (using negative distance for min-heap to act as max-heap)
        if len(nearest_neighbors) < k:
            heapq.heappush(nearest_neighbors, (-dist, i))
        else:
            # If current distance is smaller than the largest distance in heap
            if dist < -nearest_neighbors[0][0]:
                heapq.heapreplace(nearest_neighbors, (-dist, i))
    

    # Cosine similarity is the math that is taking the longest
    # Calculate distance between all training documents
    #similarities = []
    #for i, doc in enumerate(train_data):
        #sim = euclidean_distance(test_vector, doc['vector'])
        # Use negative similarity for min-heap (to get top-k largest similarities)
        #similarities.append((sim, i))

    # Get k nearest neighbors (with largest similarity)
    #k = min(k, len(train_data))  # Ensure k doesn't exceed training data size
    #nearest = heapq.nsmallest(k, similarities)
    
    # Count votes
    votes = []
    for _, idx in nearest:
        votes.append(train_data[idx]['label'])
    
    # Determine majority class
    vote_counts = Counter(votes)
    majority_class = vote_counts.most_common(1)[0][0]

    return majority_class

### Function to test kNN and return confusion matrix metrics
def test_knn(test_set, train_data, k, vocab, num_of_docs, corpus_count):
    """Test kNN classifier on test set"""
    metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for _, row in test_set.iterrows():
        true_label = row['label']
        predicted_label = predict_knn(row, train_data, k, vocab, num_of_docs, corpus_count)
        sys.stdout #buffer
        if true_label == 'True' and predicted_label == 'True':
            metrics['tp'] += 1
        elif true_label == 'False' and predicted_label == 'True':
            metrics['fp'] += 1
        elif true_label == 'False' and predicted_label == 'False':
            metrics['tn'] += 1
        elif true_label == 'True' and predicted_label == 'False':
            metrics['fn'] += 1

    return metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']

###metric outputs the passed in and derived metric values
def metric(tp, fp, tn, fn):
    print("Number of true positives:", tp)
    print("Number of true negatives:", tn)
    print("Number of false positives:", fp)
    print("Number of false negatives:", fn)
    print("Sensitivity (recall):", tp / (tp + fn))          #of real positives, how many did you catch
    print("Specificity:", tn / (tn + fp))                   #of real negatives, how many did you catch
    print("Precision:", tp / (tp + fp))                     #of real positives, how many were actually positive
    print("Negative predictive value:", tn / (tn + fn))     #of real negatives, how many were actually negative
    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))
    print("F1-score:", 2 * ( ( (tp / (tp + fp)) * (tp / (tp + fn)) ) / ( (tp / (tp + fp)) + (tp / (tp + fn)) ) ))   #mean of recall and precision


#vocab: count all words in text column without repeting words
V = 0; vocab = set()
for text in data_set['text']:
    vocab.update(set(text.split()))
V = len(vocab)

#Split data
train_set, test_set = train_test_split(data_set, TRAIN_SIZE)

###Console output (where the models are called and run and sentences is classified)
#print("Haider, Mazin, A20422384 solution:")
#print("Training set size:", TRAIN_SIZE,"%")
#if ALGO == 0:
    #print("Classifier type: Naive Bayes")
    #print("\nTraining classifier...")
    #p_false, p_true, p_word_given_false, p_word_given_true = train_naive_bayes(train_set, V)
    #print("Testing classifier...")
    #tp, fp, tn, fn = test_naive_bayes(test_set, p_false, p_true, p_word_given_false, p_word_given_true)
    #print("\nTest results / metrics:")
    #metric(tp, fp, tn, fn)

    #option = 'Y'
    #while (option[0].lower().strip() == 'y'):
        #print()
        #sentence = input("Enter your sentence/document: ")
        #sent_p_false, sent_p_true = test_naive_bayes(sentence, p_false, p_true, p_word_given_false, p_word_given_true)
        #print("\nSentence/document S:", sentence)
        #if sent_p_true > sent_p_false:
            #print("was classified as True")
        #else:
            #print("was classified as False")
        #print("P(False | S) =", sent_p_false)
        #print("P(True | S) =", sent_p_true)
        #print()
        #option = input("Do you want to enter another sentence [Y/N]? ")
#else:
    #print("Classifier type: k-NN")
    #print("\nTraining classifier...")
    
    # Train kNN (store training data)
    #train_data, corpus_count = train_knn(train_set, vocab)

    # Determine optimal k (try odd values from 1 to 21)
    #print("Finding optimal k value...")
    #best_k = 5 # default
    #best_accuracy = 0

    # Use a small validation set from training data to find optimal k
    # Split training data into train and validation (80/20)
    #val_size = int(len(train_set) * 0.2)
    #train_subset = train_set.head(len(train_set) - val_size)
    #val_subset = train_set.tail(val_size)
    
    # Train on subset 80% of the train set
    #train_subset_data, _ = train_knn(train_subset, vocab)

    #Now train_data (entire training set) and train_subset_data (80% of training set) are in b_o_w 

    # Train different k values
    #for k in range(1, 8, 2): # try odd k values from 1 to 21
        #Pass in 20% of dataset, and the 80% turned into b_o_w dataset
        #tp, fp, tn, fn, = test_knn(val_subset, train_subset_data, k, vocab, len(train_subset), corpus_count)
        #accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        #if accuracy > best_accuracy:
            #best_accuracy = accuracy
            #best_k = k
    
    #print(f"Selected k = {best_k} with validation accuracy = {best_accuracy:.4f}")

    #print("Testing classifier...")
    #tp, fp, tn, fn = test_knn(test_set, train_data, best_k, vocab, len(train_subset), corpus_count)
    #print("\nTest results / metrics:")
    #metric(tp, fp, tn, fn)

    #option = 'Y'
    #while (option[0].lower().strip() == 'y'):
        #print()
        #sentence = input("Enter your sentence/document: ")
        #predicted_class = predict_knn(sentence, train_data, best_k, vocab, len(train_subset), corpus_count)
        #print("\nSentence/document S:", sentence)
        #print(f"was classified as {predicted_class}")
        #print()
        #option = input("Do you want to enter another sentence [Y/N]?")

#vocab: count all words in text column without repeting words
V = 0; vocab = set()
for text in data_set['text']:
    vocab.update(set(text.split()))
V = len(vocab)

# Split data
train_set, test_set = train_test_split(data_set, TRAIN_SIZE)

### TEST ON SMALL SUBSET FIRST ###
print("Haider, Mazin, A20507214 solution:")
print("Training set size:", TRAIN_SIZE,"%")
print("\n" + "="*60)
print("STEP 1: Testing on small subset first...")
print("="*60)

# Test on small subset (e.g., 100 samples) to verify implementation
test_on_small_subset(data_set, ALGO, test_size=100)

# Ask user if they want to continue to full test
print("\n" + "="*60)
response = input("Small subset test complete. Continue with full dataset test? [Y/N]: ")
print("="*60)

if response.lower().strip() != 'y':
    print("Exiting program.")
    sys.exit(0)

### FULL TEST ###
print("\n" + "="*60)
print("STEP 2: Running full test on complete dataset")
print("="*60)

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
    print("\nTraining classifier...")
    
    # Train kNN (store training data)
    train_data, corpus_count = train_knn(train_set, vocab)

    # Determine optimal k (try odd values from 1 to 21)
    print("Finding optimal k value...")
    best_k = 5  # default
    best_accuracy = 0

    # Use a small validation set from training data to find optimal k
    # Split training data into train and validation (80/20)
    val_size = int(len(train_set) * 0.2)
    train_subset = train_set.head(len(train_set) - val_size)
    val_subset = train_set.tail(val_size)
    
    # Train on subset 80% of the train set
    train_subset_data, _ = train_knn(train_subset, vocab)

    # Try different k values
    print("Trying k values: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21")
    for k in range(1, 22, 2):
        tp, fp, tn, fn = test_knn(val_subset, train_subset_data, k, vocab, len(train_subset), corpus_count)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        print(f"  k={k}: accuracy = {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    print(f"\nSelected k = {best_k} with validation accuracy = {best_accuracy:.4f}")

    print("\nTesting classifier on full test set...")
    tp, fp, tn, fn = test_knn(test_set, train_data, best_k, vocab, len(train_subset), corpus_count)
    print("\nTest results / metrics:")
    metric(tp, fp, tn, fn)

    option = 'Y'
    while (option[0].lower().strip() == 'y'):
        print()
        sentence = input("Enter your sentence/document: ")
        predicted_class = predict_knn(sentence, train_data, best_k, vocab, len(train_subset), corpus_count)
        print("\nSentence/document S:", sentence)
        print(f"was classified as {predicted_class}")
        print()
        option = input("Do you want to enter another sentence [Y/N]? ")



