import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from collections import defaultdict
import re
from nltk.stem import PorterStemmer

import sys, time

from gensim.models import Word2Vec
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import RegexpTokenizer

class Helper:
    def __init__(self, set_num, file_name):
        self.readData(set_num, file_name)
        # self.readData(set_num)

    def sent_vectorizer_average(self, sent, model, dimension):
        sent_vec = np.zeros(dimension)
        for w in sent:
            sent_vec = np.add(sent_vec, model[w])

        return sent_vec / np.sqrt(sent_vec.dot(sent_vec))


    def sent_vectorizer_concatenate(self, sent, model, dimension):
        # sent_vec = np.empty([0, deminsion])
        # for w in sent:
        #     sent_vec = np.append(sent_vec, model[w].reshape(1,-1), axis=0)
        #
        # return sent_vec

        num_w = len(sent)
        sent_vec = np.zeros([num_w, dimension])
        for idx, w in enumerate(sent):
            sent_vec[idx] = model[w].reshape(1, -1)

        return sent_vec


    def readData(self, set_num, file_name='../data/training_set_rel3.tsv'):
        # training data
        print("Reading from:" + file_name)
        df = pd.read_csv(file_name, sep='\t', header = 0, encoding = "ISO-8859-1")

        tokenizer = RegexpTokenizer(r'\w+')
        porter = PorterStemmer()
        self.sent = []
        for i in range(df.shape[0]):
            sentences = re.split(r'(?<=[\.\!\?]) ', df['essay'][i])
            word_list = []
            for sent in sentences:
                sent_word_list = tokenizer.tokenize(sent)
                self.stemWordList(sent_word_list)
                self.sent.append(sent_word_list)
                word_list.extend(sent_word_list)
            # for sent in sentences:
            #     sent_word_list = tokenizer.tokenize(porter.stem(sent))
            #     self.sent.append(sent_word_list)
            #     word_list.extend(sent_word_list)
            df.at[i, 'essay'] = word_list

        dfs = []
        for i in range(8):
            dfs.append(df.loc[df['essay_set'] == i+1])

        self.sentences = dfs[int(set_num)]['essay'].values
        self.labels = dfs[int(set_num)]['domain1_score'].values

    @staticmethod
    def stemWordList(wl):
        porter = PorterStemmer()
        for idx, w in enumerate(wl):
            wl[idx] = porter.stem(w)

    def trainWord2Vec(self, embedding_size):
        # training model
        self.model = Word2Vec(self.sent, min_count=1, size=embedding_size)
         
        # get vector data
        self.X = self.model[self.model.wv.vocab]

        print('Word vector shape:', self.X.shape)

        vocab = self.model.wv.vocab.keys()


    def getPadding3D(self):
        self.trainWord2Vec()
        V = []
        for sentence in self.sentences:
            V.append(self.sent_vectorizer_concatenate(sentence, self.model, self.X.shape[1]))
        print("Number of instances: ", len(V))

        maxLength = 0
        for i in range(len(V)):
            temp = V[i].shape[0]
            if temp > maxLength:
                maxLength = temp
        print("Max essay length:", maxLength)

        # padding from
        # V_padding = np.empty((0,maxLength,self.X.shape[1])) # X is the vocabulary

        V_padding = np.zeros((len(V), maxLength, self.X.shape[1]))

        # time consuming!
        print(V_padding.shape)
        for idx in range(len(V)):
            # if V[idx].shape[0] < maxLength:
            #     padding = maxLength - V[idx].shape[0]
            #     V[idx] = np.append(np.zeros((padding, self.X.shape[1])), V[idx], axis=0)

            # V_padding = np.append(V_padding, V[art].reshape((1, maxLength, self.X.shape[1])), axis=0)
            V_padding[idx, :V[idx].shape[0]] = V[idx]

            print(V_padding.shape)

        return V_padding, maxLength, self.labels

    def get_embed(self, embedding_size):
        self.trainWord2Vec(embedding_size)
        # self.visualize()

        essays = []
        for sentence in self.sentences:
            essays.append(self.sent_vectorizer_concatenate(sentence, self.model, self.X.shape[1]))
        print("Number of instances: ", len(essays))

        maxLength = max(ess.shape[0] for ess in essays)
        print("Max essay length:", maxLength)

        # # padding from
        V_padding = np.empty((0, maxLength*self.X.shape[1])) # X is the vocabulary, X.shape[1] is the embedding size

        # time consuming!
        print(V_padding.shape)
        V = essays
        for art in range(len(V)):
            if V[art].shape[0] < maxLength:
                padding = maxLength - V[art].shape[0]
                V[art] = np.append(np.zeros((padding, self.X.shape[1])), V[art], axis=0)

            # V_padding = np.append(V_padding, V[art].reshape((1, maxLength, self.X.shape[1])), axis=0)
            V_padding = np.append(V_padding, V[art].reshape((1, -1)), axis=0)

        print(V_padding.shape)

        return V_padding, maxLength, self.labels

    # def get_embed(self, embedding_size):
    #     self.trainWord2Vec(embedding_size)
    #     essays = []
    #     for sentence in self.sentences:
    #         essays.append(self.sent_vectorizer_concatenate(sentence, self.model, self.X.shape[1]))
    #     print("Number of instances: ", len(essays))
    #
    #     maxLength = 0
    #     for i in range(len(essays)):
    #         temp = essays[i].shape[0]
    #         if temp > maxLength:
    #             maxLength = temp
    #     print("Max essay length:", maxLength)
    #
    #     return essays, maxLength, self.labels


    def getPadding2D(self):
        self.trainWord2Vec()
        V = []
        for sentence in self.sentences:
            V.append(self.sent_vectorizer_concatenate(sentence, self.model, self.X.shape[1]))
        print("Number of instances: ", len(V))

        maxLength = 0
        for i in range(len(V)):
            temp = V[i].shape[0]
            if temp > maxLength:
                maxLength = temp
        print("Max essay length:", maxLength)

        V_padding = np.empty((0, self.X.shape[1]))
        print(V_padding.shape)
        for i in range(len(V)):
            print(i)
            if V[i].shape[0] < maxLength:
                padding = maxLength - V[i].shape[0]
                V[i] = np.append(np.zeros((padding, self.X.shape[1])), V[i], axis=0)

            V_padding = np.append(V_padding, V[i].reshape((maxLength, self.X.shape[1])), axis=0)

        return V_padding, maxLength, self.labels


    def visualize(self):
        prep = ['am', 'by', 'to', 'on', 'than', 'in']
        noun = ['computer', 'time', 'people', 'friends', 'family', 'newspaper']
        verb = ['benifits', 'think', 'thinking','use', 'using', 'do', 'talk','talking', 'help', 'spend', 'spending', 'asking', 'ask']
        adj = ['great', 'tramendisly', 'powerful', 'helpful']
        porter = PorterStemmer()


        words = prep + noun + verb + adj
        counter = self.count_sort()

        # fig, ax = plt.subplots()
        # for w in words:
        #     coor = self.model[porter.stem(w)]
        #     ax.scatter(coor[0], coor[1])
        #     ax.annotate(porter.stem(w) + ' {0:d}'.format(counter[porter.stem(w)]), (coor[0], coor[1]))
        words = self.model.wv.vocab.keys()
        fig, ax = plt.subplots()
        for w in words[:100]:
            coor = self.model[porter.stem(w)]
            ax.scatter(coor[0], coor[1])
            ax.annotate(porter.stem(w), (coor[0], coor[1]))

        plt.show()

    def count_sort(self):
        porter = PorterStemmer()
        counter = defaultdict(int)
        for sent in self.sentences:
            for w in sent:
                counter[porter.stem(w)] += 1
        return counter

    # def visualize(self):
    #     prep = ['am', 'by', 'to', 'on', 'than', 'in']
    #     noun = ['computer', 'time', 'people', 'friends', 'family', 'newspaper']
    #     verb = ['benifits', 'think', 'thinking','use', 'using', 'do', 'talk','talking', 'help', 'spend', 'spending', 'asking', 'ask']
    #     adj = ['great', 'tramendisly', 'powerful', 'helpful']
    #
    #
    #     words = prep + noun + verb + adj
    #     counter = self.count_sort()
    #
    #     fig, ax = plt.subplots()
    #     for w in words:
    #         coor = self.model[w]
    #         ax.scatter(coor[0], coor[1])
    #         ax.annotate(w + ' {0:d}'.format(counter[w]), (coor[0], coor[1]))
    #
    #     plt.show()
    #
    # def count_sort(self):
    #     counter = defaultdict(int)
    #     for sent in self.sentences:
    #         for w in sent:
    #             counter[w] += 1
    #     return counter