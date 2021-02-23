import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
# nltk.download('stopwords')
from collections import Counter



google_blocks = {"block_key":[],"product_id":[]}
amazon_blocks = {"block_key":[],"product_id":[]}
google_df = pd.read_csv('google.csv')
amazon_df = pd.read_csv('amazon.csv')
stop_words = set(stopwords.words('english'))


# removes stop words from string, returns a filtered list    
def rem_stop(string):
    wordlist = nltk.word_tokenize(string)
    filteredList = [w for w in wordlist if not w in stop_words]
    return filteredList

# add all google titles to string
google_titles= ""
for i in range(len(google_df.index)):
    g_name = google_df['name'][i]
    google_titles=google_titles+" "+g_name

# remove all stop words
google_titles = rem_stop(google_titles)
# get a frequency count for each word in google data set
counted = Counter(google_titles)
#create a list of tuples with each words frequency
mydict = counted.items()
counters = list(mydict)
counters = sorted(counters, key=lambda x: x[1],)

tups = []
for tupl in counters:
    if tupl[0].isalpha():
        tups.append(tupl)
# get the 500 most frequent words
tups = tups[-500:]

for i in range(len(google_df.index)):
    #get google data
    g_id = google_df["id"][i]
    g_name = google_df['name'][i]
    name_list = g_name.split()
    for word in name_list:
        # iterate throgh title and list of words
        for tup in tups:
            block = tup[0]
            if block == word:
                google_blocks['block_key'].append(block)
                google_blocks['product_id'].append(g_id)


df = pd.DataFrame(google_blocks)
df.to_csv('google_blocks.csv',index=False)

for i in range(len(amazon_df.index)):
    #get amazon data
    a_id = amazon_df["idAmazon"][i]
    a_name = amazon_df['title'][i]
    name_list = a_name.split()
    for word in name_list:
        # iterate throgh title and list of words
        for tup in tups:
            block = tup[0]
            # add to block if word occcurs in title
            if block == word:
                amazon_blocks['block_key'].append(word)
                amazon_blocks['product_id'].append(a_id)

            

df = pd.DataFrame(amazon_blocks)
df.to_csv('amazon_blocks.csv',index=False)
