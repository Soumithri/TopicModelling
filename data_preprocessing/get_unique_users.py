import sys
from pprint import pprint
from pymongo import MongoClient


MONGO_HOST = 'localhost'
MONGO_PORT = 27017

class get_unique_users():

    def __init__(self):
        self.client = MongoClient(MONGO_HOST, MONGO_PORT)
        self.db = self.client['tweetCorpus']
        self.coll = self.db['historical_tweets']

    def get_list(self):
        query = self.coll.distinct('user.id_str')
        with open('unique_users_list2.csv','w') as outfile:
            for doc in query:
                outfile.write(doc.encode('utf-8')+'\n')
        outfile.close()


userIDs = get_unique_users()
userIDs.get_list()
