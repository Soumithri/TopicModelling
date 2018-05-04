import gensim
import logging
import pandas as pd
import re
import nltk
import pyLDAvis.gensim
from pprint import pprint
from gensim import corpora

class build_models():

    def __init__(self):
        self.colnames = {'userid', 'text'}
        # Create an empty dataframe
        self.df = pd.read_csv('./historical_tweets2/preprocessed_data_s_username.csv',
                                names=['userid', 'text'], header=None)
        self.df = self.df.dropna(subset=['text'])
        #unique_user_list = self.df.userid.unique()
        self.df = self.df.groupby('userid')['text'].apply(' '.join).reset_index()
        self.df = self.df.apply(lambda row: nltk.word_tokenize(row["text"]), axis=1)
        self.lda = gensim.models.ldamodel.LdaModel

    def main(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        documents = self.create_documents()
        pprint(len(documents))
        dictionary, dt_matrix = self.doc_term_matrix(documents)
        ldamodel = self.lda(dt_matrix, num_topics=50, id2word = dictionary, passes=50,
                            update_every=1)
        print(ldamodel.print_topics(num_topics=20, num_words=10))
        vis_data = pyLDAvis.gensim.prepare(ldamodel, dt_matrix, dictionary)
        pyLDAvis.show(vis_data)

    def doc_term_matrix(self, documents):
        # Creating the dictionary of our Corpus
        dictionary = corpora.Dictionary(documents)
        # Save dictionatary
        dictionary.save('./ldaModels/gensimLDA/tweets.dict')
        # Creating the term-document doc_term_matrix
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]
        # Storing in the DT matrix for later use
        corpora.MmCorpus.serialize('./ldaModels/gensimLDA/tweets.mm', doc_term_matrix)
        return dictionary, doc_term_matrix

    def create_documents(self):
        documents = list()
        for document in self.df:
            #print(document)
            documents.append(document)
        #print(documetns)
        return documents


if __name__=='__main__':
    model = build_models()
    model.main()
