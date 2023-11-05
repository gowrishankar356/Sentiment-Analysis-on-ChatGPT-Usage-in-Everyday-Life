#!/usr/bin/env python
# coding: utf-8

# # Topic Modelling and Sentiment analysis on ChatGPT

# In[1]:


#importing required libraries
import praw
import pandas as pd
import csv as csv
import matplotlib.pyplot as plt
import numpy as np
import time


# In[24]:


#connecting to our reddit application
reddit = praw.Reddit(client_id='5O2pjVeLTjdPuzmmwFaHpA',                      client_secret='Dx0o8h9lNreiezhBC5EKOjs9e-Yhcg',                      user_agent='MacOS:com.example.myapp:v1.0.0 (by /u/GowriShankar356)',                      username='GowriShankar356',                      password='Naruto@2398')


# # Data Collection using Reddit

# ## Data Collection using Subreddit r/ChatGPT

# In[25]:


subreddit = reddit.subreddit('ChatGPT')
submissions = subreddit.search('ChatGPT')
submissions = subreddit.hot(limit=1000)


# In[26]:


#python dictionary to store the collected data
post_dict = {"ID" : [], 
    "Title": [], 
             "Score": [], "Up vote Ratio" : [],
              "Total Comments": [], "Post URL": [], "Date": []
              }


# In[27]:


l_count_1 = 0
for submission in submissions:
    l_count_1 = l_count_1 + 1
    i = submission.title
    post_dict["ID"].append(submission.id)
    post_dict["Title"].append(submission.title)
    post_dict["Score"].append(submission.score)
    post_dict["Up vote Ratio"].append(submission.upvote_ratio)
    post_dict["Total Comments"].append(submission.num_comments)    
    post_dict["Post URL"].append(submission.url)
    post_dict["Date"].append( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submission.created_utc)))


# In[28]:


print(l_count_1)


# ## Data Collection using Subreddit r/ChatGPTPro
# 

# In[29]:


subreddit_1 = reddit.subreddit('ChatGPTPro')
submissions_1 = subreddit.search('ChatGPTPro')
submissions_1 = subreddit.hot(limit=1000)


# In[30]:


l_count_2 = 0
for submission in submissions_1:
    l_count_2 = l_count_2 + 1
    i = submission.title
    post_dict["ID"].append(submission.id)
    post_dict["Title"].append(submission.title)
    post_dict["Score"].append(submission.score)
    post_dict["Up vote Ratio"].append(submission.upvote_ratio)
    post_dict["Total Comments"].append(submission.num_comments)    
    post_dict["Post URL"].append(submission.url)
    post_dict["Date"].append( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submission.created_utc)))


# In[31]:


print(l_count_2)


# ## Data Collection using Subreddit r/ChatGPTGoneWild

# In[32]:


subreddit_2 = reddit.subreddit('ChatGPTGoneWild')
submissions_2 = subreddit_2.search('ChatGPTGoneWild')
submissions_2 = subreddit_2.hot(limit=1000)


# In[33]:


l_count_3 = 0
for submission in submissions_2:
        l_count_3 = l_count_3 + 1
        post_dict["ID"].append(submission.id)
        post_dict["Title"].append(submission.title)
        post_dict["Score"].append(submission.score)
        post_dict["Up vote Ratio"].append(submission.upvote_ratio)
        post_dict["Total Comments"].append(submission.num_comments)    
        post_dict["Post URL"].append(submission.url)
        post_dict["Date"].append( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submission.created_utc)))
        


# In[34]:


print(l_count_3)


# ## Data Collection using Subreddit r/ChatGPTCoding

# In[35]:


subreddit_3 = reddit.subreddit('ChatGPTCoding')
submissions_3 = subreddit_3.search('ChatGPTCoding')
submissions_3 = subreddit_3.hot(limit=1000)


# In[36]:


l_count_4 = 0
for submission in submissions_3:
        l_count_4 = l_count_4 + 1
        post_dict["ID"].append(submission.id)
        post_dict["Title"].append(submission.title)
        post_dict["Score"].append(submission.score)
        post_dict["Up vote Ratio"].append(submission.upvote_ratio)
        post_dict["Total Comments"].append(submission.num_comments)    
        post_dict["Post URL"].append(submission.url)
        post_dict["Date"].append( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submission.created_utc)))
        


# In[37]:


print(l_count_3)


# ## Data Collection using Subreddit r/ChatGPT3

# In[38]:


subreddit_4 = reddit.subreddit('ChatGPT_Prompts')
submissions_4 = subreddit_4.search('ChatGPT_Prompts')
submissions_4 = subreddit_4.hot(limit=1000)


# In[39]:


l_count_5 = 0
for submission in submissions_4:
        l_count_5 = l_count_5 + 1
        post_dict["ID"].append(submission.id)
        post_dict["Title"].append(submission.title)
        post_dict["Score"].append(submission.score)
        post_dict["Up vote Ratio"].append(submission.upvote_ratio)
        post_dict["Total Comments"].append(submission.num_comments)    
        post_dict["Post URL"].append(submission.url)
        post_dict["Date"].append( time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submission.created_utc)))
        


# In[40]:


print(l_count_5)


# In[41]:


#Total Number of submissions collected
print("Total Number of submissions Collected :"+str(l_count_1 + l_count_2 + l_count_3 + l_count_4 + l_count_5))


# In[42]:


#converting the collected posts dictionary into DataFrame
top_posts = pd.DataFrame(post_dict)
top_posts


# In[44]:


#saving the DataFrame as CSV file
top_posts.to_csv('data_set_fp.csv', encoding='utf-8')


# # Topic Modelling

# In[45]:


#pip install nltk


# In[46]:


#Importing necessary libraries for topic modelling
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[47]:


#importing csv data that is collected
raw_data = pd.read_csv('data_set_fp.csv')


# In[48]:


#collecting the title of submissions
title_docs = raw_data['Title']


# In[49]:


#neglecting some stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    clean_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(clean_tokens)

title_docs = title_docs.apply(clean_text)


# In[50]:


vectorizer = TfidfVectorizer(max_features=1000)
tfidf = vectorizer.fit_transform(title_docs)


# In[51]:


#creatting Latent Dirichlet Allocation rule for topic modelling
lda_2 = LatentDirichletAllocation(n_components=4,        
                                  max_iter=10,               
                                  learning_method='online',   
                                  random_state=100,          
                                  batch_size=128,            
                                  evaluate_every = -1,       
                                  n_jobs = -1 )
lda_2.fit(tfidf)


# In[52]:


print(lda_2)


# In[73]:


def print_top_words(model, feature_names, n_top_words):
    table = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        table.append([f"Topic :-- {topic_idx}", ", ".join(top_words)])
    
    # Print the table
    header = ["Topic", "Top Words"]
    print("{:<15} {}".format(header[0], header[1]))
    print("=" * 50)
    for row in table:
        print("{:<15} {}".format(row[0], row[1]))
        
print_top_words(lda_2, vectorizer.get_feature_names(), 30)


# # Sentiment Analysis using TextBlob

# In[84]:


pip install textblob


# In[3]:


from textblob import TextBlob


# In[2]:


df = pd.read_csv('data_set_fp.csv')


# In[5]:


#dropping unnecessary columns
df = df.drop(['Unnamed: 0','ID','Score','Post URL'],axis = 1)
df.tail(4)


# In[6]:


# sentiment analysis using textblob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score


# In[7]:


#creating the sentiment scores for each data point
df['Sentiment Score'] = df['Title'].apply(get_sentiment)


# In[8]:


#categorizing using sentiment score
df['Sentiment'] = df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')


# In[9]:


#For aggregating porpose
l_sentiment = []

l_postive = 0
l_neutral = 0
l_negative = 0

for i in range(0,3501):
    if df.loc[i,'Sentiment'] == 'Positive':
        l_postive = l_postive + 1
    elif df.loc[i,'Sentiment'] == 'Negative':
        l_negative = l_negative + 1
    else:
        l_neutral = l_neutral + 1
        
l_lables = ['Positive','Negative','Neutral']  


# In[10]:


l_sentiment.append(l_postive)
l_sentiment.append(l_negative)
l_sentiment.append(l_neutral)

print(l_sentiment)


# In[11]:


df.head(5)


# In[12]:


#saving the DataFrame as CSV file
df.to_csv('sentiment_analysis_textblob.csv', encoding='utf-8')


# In[13]:


#Plotting and representation of Sentiment Analysis: 
y_axis = np.array(l_sentiment)

plt.pie(y_axis , labels = l_lables, autopct='%1.1f%%',startangle = 90)
plt.title('Sentiment Analysis on ChatGPT') 
plt.show() 


# In[14]:


# Extract the year from the timestamp column
df['Year'] = pd.to_datetime(df['Date']).dt.year

# Group the data by year and sentiment polarity, and count the number of posts in each group
df_grouped = df.groupby(['Year', 'Sentiment']).size().reset_index(name='Count')

# Pivot the data to have the sentiment polarity as columns and the year as rows
df_pivot = df_grouped.pivot(index='Year', columns=['Sentiment'], values='Count')

# Plot the bar chart
df_pivot.plot(kind='bar', stacked=True)
plt.title('Sentiment polarity distribution per year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

