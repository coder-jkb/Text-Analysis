# coding: utf-8

# In[1]:


import pandas as pd
import nltk 
import pyphen # for syllables


# ## Text Extraction
# > ### web scraping (using bs4)

# In[2]:


df = pd.read_csv("Input.csv")
df.head()


# In[3]:


df.info()


# In[4]:


import requests
from bs4 import BeautifulSoup


# HTML tags to be extracted
# - title : `h1.entry-title`
# - content : `div.td-post-content`

# In[5]:


headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}
for index in df.index:
    
    # extracting HTML
    url = df['URL'][index]
    htmlContent = requests.get(url, headers=headers).content
    soup = BeautifulSoup(htmlContent, 'html.parser')

    # extracting content
    title = soup.find("h1", class_="entry-title").text
    content = soup.find("div", class_="td-post-content").text
    
    # writing in file
    file_name = str(df['URL_ID'][index])+'.txt'
    with open(file_name, 'w') as f:
        # encode into utf-8 to remove an error while scraping the text
        title = title.encode(encoding = 'utf-8')
        content = content.encode(encoding = 'utf-8')
        f.write(f'{title}\n{content}')

    # to remove this b' from byte string 
    with open(file_name,'r') as f:
        text = f.readlines()

    l1 = text[0][:-2].replace("b'","")
    l2 = text[1][:-1].replace("b'","")
    with open(file_name, 'w') as f:
        f.write(f"{l1}\n{l2}")


# ## Text Analysis

# ###  Made list of positive and negative words from given file

# In[6]:


with open('negative-words.txt','r') as f:
    words = f.read()
    neg = words.split("\n")

with open('positive-words.txt','r') as f:
    words = f.read()
    pos = words.split("\n")

dictionary = {  "positive": pos,
                "negative": neg } 


# ### Reading StopWords files, and making `list` of stop words

# In[7]:


import re
stop_words_files = ['StopWords_Generic.txt',
                    'StopWords_Names.txt',
                    'StopWords_DatesandNumbers.txt',
                    'StopWords_Auditor.txt',
                    'StopWords_GenericLong.txt',
                    'StopWords_Currencies.txt',
                    'StopWords_Geographic.txt']

with open("stop_words.txt", "w") as s:
    for file in stop_words_files:
        with open(file, 'r') as f:
            contents = f.read()
            s.write(contents)
# extracting raw text of stop words
s =  open("stop_words.txt", "r")
stop_words_txt = s.read()
s.close()

raw_stop_list = re.split(r'[| \n]\s*', stop_words_txt.lower())
stop_list = [word for word in raw_stop_list if word.isalpha()]
stop_list


# In[8]:


# code https://monkeylearn.com/blog/text-cleaning/


# ### Cleaning the text in ( `file_to_list()` ) function 

# In[9]:


# function returns cleans text and 
# returns words containig only alphabets in file in a list
# args: file_name => str
# return list
def file_to_list(file_name):
    with open(file_name,'r') as f:
        text = f.read()
        corp = re.sub('[^a-zA-Z]+',' ', text).strip()
        corp = str(corp).lower()
        tokens = nltk.word_tokenize(corp)     
        return (tokens)

# function that returns number of sentences in a file
# args: file_name => str
# returns: len(sentences) => int
def no_of_sentences(file_name):
    with open(file_name,'r') as f1:
        para = f1.read()
    sentences = nltk.sent_tokenize(para)
    return len(sentences)

# args: word => str, 
# returns : lnumbe of syllables => int
def syllables(word):
    pyp = pyphen.Pyphen(lang='en')
    syll = pyp.inserted(word)
    # print(syll)
    return( len(syll.split("-")) )


# ### Excluding the `stop_list` words and
# ### Calculaing scores and other variables for each txt file

# In[10]:


# function to calculate scores
# args: 
#   file_name => str,
#   stop => list (containing stop words), 
#   dictionary => dict (available dictionary of positive and negative words)
# returns: scores => dict
def scores(file_name, dictionary, stop_list):

    # extract list of words from the txt file
    file = file_to_list(file_name)

    # exclude words in stop file and store in stop_excluded list
    stop_excluded = set(file).difference(set(stop_list))

    # get list of elements common between file and positive
    pos_in_file = list( set.intersection(stop_excluded , 
                                        set(dictionary["positive"])) )
    # get list of elements common between file and negative
    neg_in_file = list( set.intersection(stop_excluded , 
                                        set(dictionary["negative"])))

    # We count the total cleaned words present in the text by 
    # 1. removing the stop words (using stopwords class of nltk package).
    # 2. removing any punctuations like ? ! , . from the word before counting.
    file = list(stop_excluded)
    no_of_words = len(file)

    # scores
    pos_score = len(pos_in_file)
    neg_score = len(neg_in_file)
    # Polarity Score = (Positive Score – Negative Score)
    #                                       / 
    #                        ((Positive Score + Negative Score) + 0.000001)
    polarity_score = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)

    # Subjectivity Score = (Positive Score + Negative Score)
    #                                   / 
    #                       ((Total Words after cleaning) + 0.000001)
    subjectivity_score = (pos_score + neg_score) / (no_of_words + 0.000001)

    # Average Number of Words Per Sentence (average sentence length)
    words_per_sentence = no_of_words / no_of_sentences(file_name)

    # Analysis of Readability (Gunning Fog index)
    # Average Sentence Length = the number of words / the number of sentences
    # Percentage of Complex words = the number of complex words / the number of words 
    # Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)

    # Complex Word Count
    complex_words = [w for w in file if syllables(w) > 2 ]
    percent_complex_words = len(complex_words) / no_of_words
    fog_index = 0.4 * (words_per_sentence + percent_complex_words)

    # Syllable Count Per Word
    syllables_per_word = sum([syllables(w) for w in file]) / no_of_words

    # Personal Pronouns
    personal_pronouns = []
    for i in range(len(file)):
        if file[i] in ['i', 'we', 'my', 'ours']:
            personal_pronouns.append(file[i])

        # to exclude the US (country)
        elif file[i] == 'us' and file[i-1] != 'the':
            personal_pronouns.append(file[i])

    # Average Word Length
    # Average Word Length is calculated by the formula:
    # Sum of the total number of characters in each word/Total number of words
    sum_char = 0
    for word in ["I", "am","good","boy"]:
        sum_char += len(word)

    avg_word_len = sum_char / no_of_words

    # dict of scores
    scores = {
        "POSITIVE SCORE": pos_score,
        "NEGATIVE SCORE": neg_score,
        "POLARITY SCORE": polarity_score,
        "SUBJECTIVITY SCORE": subjectivity_score,
        "AVG SENTENCE LENGTH": words_per_sentence,
        "PERCENTAGE OF COMPLEX WORDS": percent_complex_words,
        "FOG INDEX": fog_index,
        "AVG NUMBER OF WORDS PER SENTENCE": words_per_sentence,
        "COMPLEX WORD COUNT": len(complex_words),
        "WORD COUNT": no_of_words,
        "SYLLABLE PER WORD": syllables_per_word,
        "PERSONAL PRONOUNS": len(personal_pronouns),
        "AVG WORD LENGTH": avg_word_len 
    }

    return (scores)


# ### Creating the output.csv file

# In[11]:


data = []
for i in range(1,151):
    txt_file = str(i) + '.txt'
    data.append( scores(txt_file, dictionary, stop_list) )

score_df = pd.DataFrame( data )
score_df


# In[12]:


output = pd.concat([df, score_df], axis=1)
output.head()


# In[13]:


with open('output.csv', 'w') as output_file:
    output_file.write(output.to_csv(index=False, line_terminator='\n'))

