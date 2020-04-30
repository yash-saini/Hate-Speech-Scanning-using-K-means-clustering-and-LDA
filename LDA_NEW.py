# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:23:45 2020

@author: YASH
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:44:26 2020

@author: YASH
"""

import csv
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.sklearn
# Importing Gensim
#import gensim
#from gensim import corpora
'''
def LDA(tweets):
    tweets = [doc.split() for doc in tweets]
    
    dictionary = corpora.Dictionary(tweets )
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in tweets]
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel
    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
    return ldamodel
    
'''
def CLUSTER(lda_output):
     # Construct the k-means clusters
    clusters = KMeans(n_clusters=10, random_state=100).fit_predict(lda_output)
    
    # Build the Singular Value Decomposition(SVD) model
    svd_model = TruncatedSVD(n_components=2)  # 2 components
    lda_output_svd = svd_model.fit_transform(lda_output)
    
    # X and Y axes of the plot using SVD decomposition
    x = lda_output_svd[:, 0]
    y = lda_output_svd[:, 1]
    
    # Weights for the 15 columns of lda_output, for each component
    print "Component's weights: \n"
    print np.round(svd_model.components_, 2)
    
    # Percentage of total information in 'lda_output' explained by the two components
    print("Perc of Variance Explained: \n")
    print (np.round(svd_model.explained_variance_ratio_, 2))
    
    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, c=clusters)
    plt.ylabel('Component 2')
    plt.xlabel('Component 1')
    plt.title("Segregation of Topic Clusters", )
    
    
    
    
def BEST_LDA(data_vectorized,tweets,vectorizer):
    
    # Define Search Param
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(data_vectorized)
    # Best Model
    best_lda_model = model.best_estimator_
    
    # Model Parameters
    print "Best Model's Params: "
    print (model.best_params_)
    
    # Log Likelihood Score
    print "Best Log Likelihood Score: "
    print (model.best_score_)
    
    # Perplexity
    print "Model Perplexity: "
    print (best_lda_model.perplexity(data_vectorized))
    ''' n_topics = [10, 15, 20, 25, 30]
    log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.cv_results_ if gscore.params['learning_decay']==0.5]
    log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.cv_results_ if gscore.params['learning_decay']==0.7]
    log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.cv_results_ if gscore.params['learning_decay']==0.9]

    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, log_likelyhoods_5, label='0.5')
    plt.plot(n_topics, log_likelyhoods_7, label='0.7')
    plt.plot(n_topics, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()
    ''' 
    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
    
    # index names
    docnames = ["Doc" + str(i) for i in range(len(tweets))]
    
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_o, 2), columns=topicnames, index=docnames)
    
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    print ("The Dominant Topic is: \n")
    print df_document_topic
    df_document_topic.to_csv("Dominant_Topic.csv")
    
    ''' TOPIC DISTRIBUTION :- NO of documents per topic number'''
    
    print ("\n\n TOPIC DISTRIBUTION :- NO of documents per topic number \n\n")
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    print df_topic_distribution
    df_topic_distribution.to_csv("Topic_Distribution.csv")
    
    # In Jupyter notebook for better visualization
    '''panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
    print (panel)
    '''
    
    
    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)
    
    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames
    
    # View
    print "\n\n TOPIC KEYWORDS \n\n"
    print df_topic_keywords
    df_topic_keywords.to_csv("Topic_Keywords.csv")
    return best_lda_model
    
    
    
def show_topics(vectorizer, lda_model, n_words=20):
    """TOP 15 keywords for each topic"""
    print ("\n\n****TOP 15 keywords for each topic****\n\n")
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


    
def LDA_SK(data_vectorized,vectorizer):
    #Build LDA Model
    '''lda_model = LatentDirichletAllocation(n_topics=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
    '''
    lda_model = LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7,
             learning_method='online', learning_offset=10.0,
             max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
             n_components=10, n_jobs=-1, n_topics=10, perp_tol=0.1,
             random_state=100, topic_word_prior=None,
             total_samples=1000000.0, verbose=0)

    lda_output = lda_model.fit_transform(data_vectorized)

    #print(lda_model)  # Model attributes
    
    
    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(data_vectorized))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(data_vectorized))
    return lda_output
    
    
  


c=raw_input("Enter the name of the tweet file:")
c_f="POS_tagged_"+c+".csv"
db=pd.read_csv(c_f)
db.dropna(inplace=True)
tweets=db["POS_Tweet"].values
tweets= list( dict.fromkeys(tweets))
cv = CountVectorizer( analyzer='word',min_df=2,token_pattern='[a-zA-Z0-9]{3,}',max_features = 3000)
X = cv.fit_transform(tweets)
# Materialize the sparse data
data_dense = X.todense()
u=sum(np.array(data_dense > 0))
#Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", (sum(u)/float(np.size(data_dense)))*100, "%")


lda_o=LDA_SK(X,cv)
best_lda=BEST_LDA(X,tweets,cv)
'''alpha=LDA(X)
print(alpha.print_topics(num_topics=3, num_words=3))
'''

topic_keywords = show_topics(vectorizer=cv, lda_model=best_lda, n_words=10)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
print ("\n \n TOPIC KEYWORDS LDA \n\n")
print (df_topic_keywords)
df_topic_keywords.to_csv("topic_keywords_LDA_NEW.csv")



CLUSTER(lda_o)
