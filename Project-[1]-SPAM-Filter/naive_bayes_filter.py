# Import libraries
import numpy as np
import pandas as pd
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re

# Using nltk for lemmanization, stemming, and tokenization
# Using re for string replacement for single message only
# Using sckit CountVectorizer for converting to vector data

#nltk libary downloads
nltk.download('stopwords')
nltk.download('corpus')
nltk.download('punkt')
nltk.download('wordnet')
class NaiveBayesFilter:
    def __init__(self, test_set_path):
        self.vocabulary = None
        self.training_set= None
        self.test_set = None
        self.p_spam = None
        self.p_ham = None
        self.n_spam = None
        self.n_ham = None
        self.n_word_given_spam = None
        self.n_word_given_ham = None
        self.n_vocabulary = None
        self.test_set_path = test_set_path

        self.stop_words = set(stopwords.words('english'))
        self.lemmatization = None
        self.stemming = None
        self.vectorizer = None
        self.dictionary = None
        self.dictionaryDF = None
        self.spamData = None
        self.hamData = None

    def read_csv(self):
        self.training_set = pd.read_csv('train.csv', sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')
        self.test_set = pd.read_csv(self.test_set_path, sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')

    def data_cleaning(self):
        # Normalization
        # Replace addresses (hhtp, email), numbers (plain, phone), money symbols
        #print(self.training_set['v2'].head(10))
        #self.training_set['v2'] = self.training_set['v2'].str.replace(r'\b(?:https?|ftp):\/\/\S+|www\.\S+|[\w\.-]+@[\w\.-]+\.\w+\b', '', regex=True)
        #self.training_set['v2'] = self.training_set['v2'].str.replace(r'\d+', '', regex=True)
        #self.training_set['v2'] = self.training_set['v2'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
        self.training_set['v2'] = self.training_set['v2'].str.replace('\W', ' ', regex=True)
        #self.training_set['v2'] = self.training_set['v2'].str.replace(r'\$\d+(?:\.\d{2})?', '', regex=True)
        #print(self.training_set['v2'].head(10))

        # Remove the stop-words
        # Using the NLTK libary to deal with stop words
        self.training_set['v2'] = self.training_set['v2'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in self.stop_words]))
        #print(self.training_set['v2'].head(10))

        # Lemmatization - Graduate Students
        self.lemmatization = nltk.stem.WordNetLemmatizer()
        self.training_set['v2'] = self.training_set['v2'].apply(lambda x: ' '.join([self.lemmatization.lemmatize(word, pos='v') for word in word_tokenize(x)]))

        #print(self.training_set['v2'].head(10))

        # Stemming - Gradutate Students
        self.stemming = nltk.PorterStemmer()
        self.training_set['v2'] = self.training_set['v2'].apply(lambda x: ' '.join([self.stemming.stem(word.lower()) for word in word_tokenize(x)]))

        # Tokenization
        # Already tokenzied
        #self.training_set['v2'] = self.training_set['v2'].apply(lambda x: nltk.word_tokenize(x))
        #print(self.training_set['v2'].head(10))

        # Vectorization
        # Using scikit-learn's count vectorizer
        self.vectorizer = CountVectorizer()
        vectorData = self.vectorizer.fit_transform(self.training_set['v2'])
        self.vocabulary = self.vectorizer.get_feature_names_out()
        #self.vocabulary = [word for word in self.vocabulary if vectorData[:, self.vectorizer.vocabulary_[word]].sum() >= 2]
        #print(self.vocabulary)
        #print(vectorData)

        # Remove duplicates - Can you think of any data structure that can help you remove duplicates?

        # Create the dictionary
        self.dictionary = {word: index for index, word in enumerate(self.vocabulary)}
        #print(self.dictionary)
        # Convert to dataframe
        #print(vectorData)
        self.dictionaryDF = pd.DataFrame(vectorData.toarray(), columns=self.vocabulary)

        #print(self.dictionaryDF)

        self.training_set = pd.concat([self.training_set, self.dictionaryDF], axis=1)
        #print(self.training_set)

        # Separate the spam and ham dataframes
        #print(self.training_set['v1'])
        self.spamData = self.training_set[self.training_set['v1'] == 'spam']
        self.hamData = self.training_set[self.training_set['v1'] == 'ham']
        #print(self.spamData)
        #print(self.hamData)
        pass

    def calculateWords(self, df):
        return sum([len(row.split()) for row in df['v2']])

    def fit_bayes(self):
        # Calculate P(Spam) and P(Ham)
        self.p_spam = len(self.spamData) / len(self.training_set)
        self.p_ham = len(self.hamData) / len(self.training_set)

        # Calculate Nspam, Nham and Nvocabulary
        self.n_spam = self.calculateWords(self.spamData)
        self.n_ham = self.calculateWords(self.hamData)
        self.n_vocabulary = len(self.dictionaryDF)
        #print(self.n_spam)
        #print(self.n_ham)
        #print(self.n_vocabulary)
        # Laplace smoothing parameter
        alpha = .15

        # Calculate P(wi|Spam) and P(wi|Ham)
        # Creating a dictinonary that keeps track of the amount of times a word appears
        self.p_wordSpam = {word:0 for word in self.dictionaryDF['word']}
        self.p_wordHam = {word:0 for word in self.dictionaryDF['word']}
        #print(self.spamData)

        for word in self.dictionary:
                self.n_word_given_spam = self.spamData[word].sum()
                #print(self.n_word_given_spam)
                self.n_word_given_ham = self.hamData[word].sum()

                self.p_wordSpam[word] = (self.n_word_given_spam + alpha) / (self.n_spam + alpha * self.n_vocabulary)
                self.p_wordHam[word] = (self.n_word_given_ham + alpha) / (self.n_ham + alpha * self.n_vocabulary)

        #print(self.p_wordSpam)

    def train(self):
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()

    def messageCleaning(self, message):
        # Replaces Addresses and other junk information
        # We are using re sub for this as it's just a single string
        #message = re.sub(r'\b(?:https?|ftp):\/\/\S+|www\.\S+|[\w\.-]+@[\w\.-]+\.\w+\b', '', message)
        #message = re.sub(r'\d+', '', message)
        #message = re.sub(r'[^a-zA-Z\s]', ' ', message)
        message = re.sub(r'\W', ' ', message)
        #message = re.sub(r'\$\d+(?:\.\d{2})?', '', message)

        # remove stop words
        words = word_tokenize(message)
        message = ' '.join([word for word in words if word.lower() not in self.stop_words])

        # Lemmatization
        message = ' '.join([self.lemmatization.lemmatize(word, pos='v') for word in word_tokenize(message)])

        # Stemming
        message = ' '.join([self.stemming.stem(word.lower()) for word in word_tokenize(message)])

        #Tokenization
        message = word_tokenize(message)

        return message

    
    def sms_classify(self, message):
        '''
        classifies a single message as spam or ham
        Takes in as input a new sms (w1, w2, ..., wn),
        performs the same data cleaning steps as in the training set,
        calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn),
        compares them and outcomes whether the message is spam or not.
        '''
        message = self.messageCleaning(message)

        p_spam_given_message = self.p_spam
        p_ham_given_message = self.p_ham

        #print(message)
        #print(self.p_ham)
        #print(self.p_wordSpam)

        for word in message:
            if word in self.p_wordSpam:
                p_spam_given_message *= self.p_wordSpam[word]
            if word in self.p_wordHam:
                p_ham_given_message *= self.p_wordHam[word]

        #print('ham probability', p_ham_given_message)
        #print('spam probability', p_spam_given_message)

        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        else:
            return 'needs human classification'

        # if p_ham_given_message > p_spam_given_message:
        #     return 'ham'
        # elif p_spam_given_message > p_ham_given_message:
        #     return 'spam'
        # else:
        #     return 'needs human classification'
        pass

    def classify_test(self):
        '''
        Calculate the accuracy of the algorithm on the test set and returns 
        the accuracy as a percentage.
        '''

        accuracy = 0
        correct = 0
        self.train()
        # print(self.test_set)
        for index, row in self.test_set.iterrows():
            classification = self.sms_classify(row['v2'])
            if classification == row['v1']:
                correct += 1
            #else:
                #print(row['v1'] + ' ' + row['v2'])
                #print(self.messageCleaning(row['v2']))
        accuracy = (correct / len(self.test_set)) * 100
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--test_dataset', type=str, default = "test.csv", help='path to test dataset')
    args = parser.parse_args()
    classifier = NaiveBayesFilter(args.test_dataset)
    acc = classifier.classify_test()
    print("Accuracy: ", acc)
