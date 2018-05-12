import gensim
import logging
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from pprint import pprint
from gensim import corpora, models

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def loadLDAModel():
    dictionary = corpora.Dictionary.load('./ldaModels/gensimLDA20/tweets.dict')
    mm_corpus = corpora.MmCorpus('./ldaModels/gensimLDA20/tweets.mm')
    #pprint(mm_corpus)
    outfile = open('./ldaModels/perplexity1.txt','w')
    trained_models = dict()
    #perplexity = list()

    # split into train and test - random sample, but preserving order
    train_size = int(round(len(mm_corpus)*0.8))
    train_index = sorted(random.sample(xrange(len(mm_corpus)), train_size))
    test_index = sorted(set(xrange(len(mm_corpus)))-set(train_index))
    train_corpus = [mm_corpus[i] for i in train_index]
    test_corpus = [mm_corpus[j] for j in test_index]

    # train_set = list()
    # test_set = list()
    # for i in range(0, len(documents-1))
    # dictionary = corpora.Dictionary(documents)
    # dictionary.save('./ldaModels/lda_train/tweets.dict')
    # # Creating the term-document doc_term_matrix
    # mm_corpus = [dictionary.doc2bow(doc) for doc in documents]
    # # Storing in the DT matrix for later use
    # corpora.MmCorpus.serialize('./ldaModels/lda_train/tweets.mm', mm_corpus)

    for num_topics in range(10, 101, 10):
        print("Training LDA(k=%d)" % num_topics)

        lda = models.ldamodel.LdaModel(corpus=train_corpus, id2word=dictionary, num_topics=num_topics,
                                update_every=1, chunksize=10000, passes=10, alpha='auto')
        # lda = models.LdaMulticore(
        #     corpus=mm_corpus, id2word=dictionary, num_topics=num_topics, workers=None,
        #     alpha='asymmetric',  # shown to be better than symmetric in most cases
        # )
        trained_models[num_topics] = lda
        #print('Perplexity: ', lda.bound(mm_corpus))
        #perplexity[num_topics].append(lda.bound(mm_corpus))
        #outfile.write('Perplexity for ' + str(num_topics) + ' topics: ' + str(lda.bound(mm_corpus)) + '\n')
        outfile.write('Perplexity for ' + str(num_topics) + ' topics: ' + str(lda.log_perplexity(test_corpus)) + '\n')
    outfile.close()
    return trained_models

def visualize():
    df = pd.read_csv('lda/metrics/perplexity1.csv', names=['K', 'Perplexity'], header=None)
    print(df)
    plt.figure()
    df.plot(x='K', y='Perplexity')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('Perplexity')
    plt.title('Number of Topics(K) Vs Perplexity')
    plt.savefig('lda/plots/TopicsVsPerplexity.png')
    plt.show()
    plt.close()
    # df = pandas.DataFrame(perplexity)
    # ax = plt.figure(figsize=(7, 4), dpi=300).add_subplot(111)
    # df.iloc[1].transpose().plot(ax=ax,  color="#254F09")
    # plt.xlim(parameter_list[0], parameter_list[-1])
    # plt.ylabel('Perplexity')
    # plt.xlabel('topics')
    # plt.title('')
    # plt.savefig('gensim_multicore_i10_topic_perplexity.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    # df.to_pickle(data_path + 'gensim_multicore_i10_topic_perplexity.df')

def save_models(named_models):
    # home = os.path.expanduser('~/')
    # models_dir = os.path.join(home, 'ldaModels')
    #
    for num_topics, model in named_models.items():
        model_path = str('./lda_train/lda_tweets_k%d.lda' % num_topics)
        model.save(model_path, separately=False)

if __name__ == '__main__':
    #models = loadLDAModel()
    #save_models(models)
    visualize()
