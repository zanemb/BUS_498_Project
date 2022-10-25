import sqlite3
import pandas as pd
import re

# import praw
# reddit = praw.Reddit(client_id='08D24AM5aG_pj3_J4DFN1Q', # App ID
#                      client_secret='lu3C66Ps1r1lNRt1D0Lf9RTgCsxz4w', # App password
#                      user_agent='Text Miner 1.0', # App name (can be different)
#                      username='Separate_Objective78', # Username
#                      password='Awesome5') # User password



def main():
    import numpy as np

    # comments
    df1 = pd.read_csv('comments_table.csv', encoding = 'utf-8')

    comments_conn = sqlite3.connect('comments.db')
    # df1.to_sql('comments_df', comments_conn) # CREATED THE SQL DATABASE

    # threads
    df2 = pd.read_csv('threads_table.csv', encoding = 'utf-8')

    threads_conn = sqlite3.connect('threads.db')
    # df2.to_sql('threads_df', threads_conn) # CREATED THE SQL DATABASE


    # ## CREATE THE TABLES 'threads' AND 'comments' ##

    # conn = sqlite3.connect('Netflix.db')
    # cur = conn.cursor()


    # delete_sql1 = """DROP TABLE threads;"""
    # delete_sql2 = """DROP TABLE comments;"""
    # create_sql1 = """CREATE TABLE threads (ID text primary key, title text, author text, url text, 
    #                 created_utc real, num_comments integer, score integer);"""
    # create_sql2 = """CREATE TABLE comments (ID text primary key, thread_ID text,
    #                 body text, author text, created_utc real, score integer,
    #                 foreign key (thread_ID) references threads(ID));"""

    # cur.execute(delete_sql1)
    # cur.execute(delete_sql2)
    # cur.execute(create_sql1)
    # cur.execute(create_sql2)


    # ## INSERT DATA RECORDS ##

    # # Technology Subreddit
    # thread_num = 0
    # subreddit1 = reddit.subreddit('technology')
    # for thread in subreddit1.search(query='netflix'):
    #     sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
    #     cur.execute(sql, (thread.id, thread.title, thread.author.name, 
    #                       thread.url, thread.created_utc, thread.num_comments, thread.score))
    #     thread.comments.replace_more(limit = 0)

    #     thread_num += 1
    #     if(thread_num % 10 == 0):
    #         print("{0} netflix thread files have been processed.".format(thread_num))

    #     comment_num = 0
    #     for comment in thread.comments:
    #         comment_num += 1
    #         # if comment_num > 10:
    #         #     break
    #         if comment.author != None:
    #             sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
    #             cur.execute(sql, (comment.id, comment.link_id, comment.body, 
    #                             comment.author.name, comment.created_utc, comment.score))

    # thread_ids_lst = []
    # thread_ids = cur.execute("SELECT ID FROM threads")
    # for id in thread_ids.fetchall():
    #     thread_ids_lst.append(''.join(id))

    # thread_num = 0
    # for thread in subreddit1.search(query='nflx'):
    #     # NOTE: there are only 3 matches!
    #     if thread.id not in thread_ids_lst:
    #         sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
    #         cur.execute(sql, (thread.id, thread.title, thread.author.name, 
    #                         thread.url, thread.created_utc, thread.num_comments, thread.score))
    #         thread.comments.replace_more(limit = 0)

    #         thread_num += 1
    #         if(thread_num % 10 == 0):
    #             print("{0} nflx thread files have been processed.".format(thread_num))

    #         comment_num = 0
    #         for comment in thread.comments:
    #             comment_num += 1
    #             # if comment_num > 10:
    #             #     break
    #             if comment.author != None:
    #                 sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
    #                 cur.execute(sql, (comment.id, comment.link_id, comment.body, 
    #                                 comment.author.name, comment.created_utc, comment.score))

    # # Stocks Subreddit
    # thread_num = 0
    # subreddit2 = reddit.subreddit('stocks')
    # for thread in subreddit2.search(query='netflix'):
    #     sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
    #     cur.execute(sql, (thread.id, thread.title, thread.author.name, 
    #                       thread.url, thread.created_utc, thread.num_comments, thread.score))
    #     thread.comments.replace_more(limit = 0)

    #     thread_num += 1
    #     if(thread_num % 10 == 0):
    #         print("{0} netflix thread files have been processed.".format(thread_num))

    #     comment_num = 0
    #     for comment in thread.comments:
    #         comment_num += 1
    #         # if comment_num > 10:
    #         #     break
    #         if comment.author != None:
    #             sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
    #             cur.execute(sql, (comment.id, comment.link_id, comment.body, 
    #                             comment.author.name, comment.created_utc, comment.score))

    # thread_ids_lst = []
    # thread_ids = cur.execute("SELECT ID FROM threads")
    # for id in thread_ids.fetchall():
    #     thread_ids_lst.append(''.join(id))

    # thread_num = 0
    # for thread in subreddit2.search(query='nflx'):
    #     if thread.id not in thread_ids_lst:
    #         sql = "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?);"
    #         cur.execute(sql, (thread.id, thread.title, thread.author.name, 
    #                         thread.url, thread.created_utc, thread.num_comments, thread.score))
    #         thread.comments.replace_more(limit = 0)

    #         thread_num += 1
    #         if(thread_num % 10 == 0):
    #             print("{0} nflx thread files have been processed.".format(thread_num))

    #         comment_num = 0
    #         for comment in thread.comments:
    #             comment_num += 1
    #             # if comment_num > 10:
    #             #     break
    #             if comment.author != None:
    #                 sql = "INSERT INTO comments VALUES (?, ?, ?, ?, ?, ?);"
    #                 cur.execute(sql, (comment.id, comment.link_id, comment.body, 
    #                                 comment.author.name, comment.created_utc, comment.score))

    # print("Done creating database!")

    # # CREATE CSV ##
    # query_threads_sql = "SELECT * FROM threads;"
    # threads_db = pd.read_sql(query_threads_sql, conn)
    # threads_db.to_csv('threads_table.csv', encoding='utf-8')

    # query_comments_sql = "SELECT * FROM comments;"
    # comments_db = pd.read_sql(query_comments_sql, conn)
    # comments_db.to_csv('comments_table.csv', encoding='utf-8')

    ## ADD RESULTS TO LISTS ##

    thread_review_dict = {}
    comment_review_list = []

    threads_cur = threads_conn.cursor()
    threads_cur.execute("SELECT ID, title FROM threads_df;")

    for row in threads_cur.fetchall():
        thread_review_dict[row[0]] = row[1]

    comments_cur = comments_conn.cursor()
    comments_cur.execute("SELECT thread_ID, body FROM comments_df;")

    count = 0
    for row in comments_cur.fetchall():
        count += 1
        comment_review_list.append((row[0].replace('t3_', ''), row[1]))

    def remove_chars(text):

        p = re.compile(r"\)\.*\,*|\.{2,}|\$+\w*\.*|[0-9]+|%\?*\,*\.*\+*|\+\'*\,*|--")
        text = p.sub('', text)

        p = re.compile(r"\((\w+)|\-(\w+)|\"(\w+)|\>+(\w+)")
        text = p.sub(r'\1', text)

        return text

    # Expanding contraction
    contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                        "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", 
                        "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", 
                        "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  
                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", 
                        "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  
                        "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
                        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                        "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                        "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 
                        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 
                        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
                        "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 
                        "she'll've": "she will have", "she's": "she is", "should've": "should have", 
                        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                        "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 
                        "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", 
                        "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                        "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 
                        "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", 
                        "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", 
                        "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", 
                        "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
                        "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", 
                        "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", 
                        "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", 
                        "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
                        "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                        "you'll've": "you will have", "you're": "you are", "you've": "you have"}

    def _get_contractions(contraction_dict):
        contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contraction_re

    contractions, contractions_re = _get_contractions(contraction_dict)

    def replace_contractions(text):  
        def replace(match):
            return contractions[match.group(0)]
        return contractions_re.sub(replace, text)

    # tokenization
    from nltk.tokenize import RegexpTokenizer

    def tokenize(text):
        tokenizer = RegexpTokenizer(r'[\w\']+|\$[\d\.]+|\S+')
        return tokenizer.tokenize(text)


    from nltk.corpus import wordnet

    def get_wordnet_pos(treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        elif treebank_tag.startswith('P'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # lemmatization
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag

    def lemmatize(text):
        wn_lt = WordNetLemmatizer()
        text = [wn_lt.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in pos_tag(text)]
        return text


    # remove 1-letter words and lengthy words of length greater than 15
    def rm_words_len(tokens):
        clean_tokens = []
        for token in tokens:
            if(len(token) > 1 and len(token) <= 20):
                clean_tokens.append(token)
        return clean_tokens

    def rm_words(tokens):
        companies = ['amazon', 'roku', 'samsung', 'google', 'youtube', \
            'disney', 'whatsapp', 'meta', 'facebook']
        clean_tokens = []
        for token in tokens:
            if token not in companies:
                clean_tokens.append(token)
        return clean_tokens

    # stop words
    from nltk.corpus import stopwords
    def rv_stopwords(tokenized_text):

        nltk_sw = stopwords.words('english')

        sw_list = [",", ".", "'", '"', '&', '', '%', '>', '’s', '’t', '’m', '’re', '’ll', '->', '?"', '."', '&#xb', '&#xb;']
        sw_list.extend(nltk_sw)
        return [word for word in tokenized_text if word not in sw_list]

    def preprocess(text):
        text = remove_chars(text)
        text = replace_contractions(text)
        text = text.lower()
        tokens = tokenize(text)
        tokens = rv_stopwords(tokens)
        tokens = rm_words_len(tokens)
        # tokens = rm_words(tokens)
        tokens = lemmatize(tokens)
        return tokens


    # Adding in a specific topic #
    def topic(thread_review_list, comment_review_list):
        p1 = re.compile(r"^ads?[\s\.\?\!\,\-\_]|[\s\.\?\!\,\-\_]ads?[\s\.\?\!\,\-\_]|[\s\.\?\!\,\-\_]ads?$\
            |^advertisements?[\s\.\?\!\,\-\_]|[\s\.\?\!\,\-\_]advertisements?[\s\.\?\!\,\-\_]|[\s\.\?\!\,\-\_]advertisements?$", re.IGNORECASE)

        p2 = re.compile(r"^commercials?[\s\.\?\!\,\-\_]|[\s\.\?\!\,\-\_]commercials?[\s\.\?\!\,\-\_]|[\s\.\?\!\,\-\_]commercials?$", re.IGNORECASE)

        id_list = []
        new_review_list = []

        for key in thread_review_list:
            # if(p1.search(thread_review_list[key]) != None) or (p2.search(thread_review_list[key]) != None):
                id_list.append(key)
                new_review_list.append(thread_review_list[key])

        for line in comment_review_list:
            # if(p1.search(line[1]) != None) or (p2.search(line[1]) != None) or line[0] in id_list:
                new_review_list.append(line[1])
        return new_review_list

    review_list = topic(thread_review_dict, comment_review_list)

    corpus = []

    for line in review_list:
        corpus.append(preprocess(line))
    print("Finished preprocessing!")




    ## SENTIMENT ANALYSIS ##
    import gensim.models

    # doc2vec
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(documents, dbow_words=1, vector_size=200, window=5, min_count=1, workers=4)


    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob

    analyzer = SentimentIntensityAnalyzer()

    lines = []
    for line in corpus:
        sentence = []
        for word in line:
            sentence.append(word)
        sentence = ' '.join(sentence)
        lines.append(sentence)
        

    # Sentiment analysis function for TextBlob tools
    def text_blob_sentiment(review, sub_entries_textblob):
        analysis = TextBlob(review)
        if analysis.sentiment.polarity > 0:
            sub_entries_textblob['positive'] = sub_entries_textblob['positive'] + 1
            return 'Positive'
        else:
            sub_entries_textblob['negative'] = sub_entries_textblob['negative'] + 1
            return 'Negative'

    sia = SentimentIntensityAnalyzer()

    def nltk_sentiment(review, sub_entries_nltk):
        vs = sia.polarity_scores(review)
        if vs['pos'] - vs['neg'] > 0:
            sub_entries_nltk['positive'] = sub_entries_nltk['positive'] + 1
            return 'Positive'
        else:
            sub_entries_nltk['negative'] = sub_entries_nltk['negative'] + 1
            return 'Negative'


    sub_entries_textblob = {'negative': 0, 'positive' : 0}
    sub_entries_nltk = {'negative': 0, 'positive' : 0}


    # when an entry is scored the same (positive, positive) by both textblob and vader, 
    # add that entry to the "true" sentiment list
    true_sent = {}
    true_sent_count = 0
    all_entries = 0

    negative_sent = 0
    positive_sent = 0

    for submission in lines:
        sent1 = text_blob_sentiment(submission, sub_entries_textblob)
        sent2 = nltk_sentiment(submission, sub_entries_nltk)
        all_entries += 1
        if sent1 == sent2:
            true_sent[submission] = sent1
            true_sent_count += 1
            if sent1 == 'Negative':
                negative_sent += 1
            else:
                positive_sent += 1

    print()
    print(f"Text Blob: {sub_entries_textblob}")
    print(f"Vader: {sub_entries_nltk}")
    print(f"Models identified {true_sent_count} true sentiments out of {all_entries}\n")
    print(f"Positive True Sentiments: {positive_sent}")
    print(f"Negative True Sentiments: {negative_sent}")

    # use the true sentiment list for the supervised learning model
    # split the true sentiment list randomly 20% to training model and 80% to the testing model

    # convert sentiment dictionary to two lists

    corpus_raw = []
    labels = []
    ts_count = 0

    positive_lst = []
    negative_lst = []

    for key in true_sent:
        corpus_raw.append(key)
        if true_sent[key] == "Positive":
            positive_lst.append(key)
            labels.append(1)
        else:
            negative_lst.append(key)
            labels.append(0)
        ts_count += 1
        if ts_count == round(len(true_sent)/5):
            break
    # print(labels)
    print(f"Training set length: {len(corpus_raw)}")

    ## SUPERVISED LEARNING ##
    # X: document vectors
    # Y: labels

    X_train = []

    for i in range(len(corpus_raw)):
        X_train.append(model.dv[i])

    X_train = np.asarray(X_train)

    Y_train = np.asarray(labels)


    # Logistic Regression #
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn import svm

    kernel = 'poly'
    # clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
    clf = svm.SVC(kernel=kernel).fit(X_train, Y_train)


    # Testing set
    test_labels = []
    test_corpus = []
    test_corpus_raw = []

    for key in true_sent:
        if key not in corpus_raw:
            test_corpus_raw.append(key)
            test_corpus.append(list(key))
            if true_sent[key] == "Positive":
                positive_lst.append(key)
                test_labels.append(1)
            else:
                negative_lst.append(key)
                test_labels.append(0)


    # print(test_labels)
    print(f"Testing set length: {len(test_corpus)}\n")

    # test document vectors
    X_test = []
    for i in range(len(test_corpus)):
        X_test.append(model.infer_vector(test_corpus[i]))

    X_test = np.asarray(X_test)

    Y_test = np.asarray(test_labels)


    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report


    Y_pred = clf.predict(X_test)
    # print(Y_pred)

    # confusion matrix, precision, recall, f1
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()

    print(confusion_matrix(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))









    # TOPIC MODELING ##

    #lsa
    from gensim import corpora, models

    new_corpus = []
    for lst in corpus:
        new_lst = []
        for word in lst:
            if word != 'netflix' and word != 'ad':
                new_lst.append(word)
        new_corpus.append(new_lst)

    dictionary = corpora.Dictionary(new_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in new_corpus]


    # from gensim.models.coherencemodel import CoherenceModel

    # coherence = CoherenceModel(model=lda_model, texts=corpus, dictionary=dictionary, coherence = "c_v")
    # print(coherence.get_coherence())

    # x = []
    # y = []
    # for k in range(3, 51):
    #     lda_model = models.LdaMulticore(bow_corpus, num_topics=k, id2word=dictionary, passes=10, workers=2)
    #     coherence = CoherenceModel(model=lda_model, texts=corpus, dictionary=dictionary, coherence = "c_v")
    #     x.append(k)
    #     y.append(coherence.get_coherence())

    # import matplotlib.pyplot as plt

    # plt.plot(x, y)
    # plt.title('Coherence vs Topic')
    # plt.xlabel('# Topics')
    # plt.ylabel('Coherence Score')
    # plt.show()


    # plt.plot(x[:20], y[:20])
    # plt.title('Coherence vs Topic')
    # plt.xlabel('# Topics')
    # plt.ylabel('Coherence Score')
    # plt.show()


    # k = 6 is the optimal number for this dataset, based off the coherence graphs
    lda_model = models.LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=10, workers=2)


    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {0} \n Words: {1}".format(idx, topic))
        print("\n")

    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    import pyLDAvis.sklearn
    lda_viz = gensimvis.prepare(lda_model, bow_corpus, dictionary)
    pyLDAvis.save_html(lda_viz, 'all_Visualization.html')




if __name__ == '__main__':
    main()