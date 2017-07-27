import matplotlib
matplotlib.use('Agg')

'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''



import os
# import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
# from wordcloud import WordCloud
from collections import Counter
from sklearn.decomposition import TruncatedSVD, PCA


'''
needed to change col name for all_text and get the beer names from the index of the df beers from pickle
'''

colors = ['c','midnightblue','purple','green','.2']

drop_lst = ['aroma', 'note', 'notes', 'sl',
            'slight', 'light', 'hint', 'bit',
            'little', 'lot', 'touch', 'character',
            'some', 'something', 'retro', 'thing',
            ' ', 'aaron', 'low', 'smell', 'smells',
            'lite', 'lost', 'end', 'clean', 'pretty',
            'faint','nice','finishes','finish','faint',
            'minimal','just', 'way', 'big', 'good',
            'uniform','grady','really','fairly','quickly',
            'starts','hue','moves']

class NewBeer(object):
    def __init__(self):
        np.random.seed(42)

    def Fit(self,read_file):
        beers = pd.read_pickle(read_file)
        self.df = beers
        text = beers['all_text'].values
        self.text = [' '.join(quality) for quality in text]

        self.beer_names = np.array(beers.index.tolist())

    def Transform(self, k = 5):
        self.k = k
        self.W, self.H, self.nmf, self.tfidf, self.top_words = self.Quick_nmf(k = self.k, print_tops = False, stop_words = drop_lst)
        self.cluster_labels = np.argmax(self.W,axis = 1)

        # get the top words associated with each beer
        beer_top_terms = []
        for beer in self.text:
            word_lst = beer.split()
            word_counter = Counter([word for word in word_lst if word in self.bag_of_words])
            words,nums = zip(*word_counter.most_common())
            i_bigger = np.argwhere(np.array(nums) > 1).flatten()
            beer_terms = np.array(words)[i_bigger]
            beer_top_terms.append(beer_terms)
        self.beer_top_terms = beer_top_terms

    def Quick_nmf(self, k = 5, top = 10, tfidf = None, print_tops = True, stop_words =[]):

        text = self.text
        labels = self.beer_names

        if tfidf == None:
            stopwords = set(list(ENGLISH_STOP_WORDS) + stop_words)
            tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=.8, min_df=.2, stop_words = stopwords, max_features = 10000, )

        X = tfidf.fit_transform(text)
        bag = np.array(tfidf.get_feature_names())
        self.bag_of_words = bag

        nmf = NMF(n_components = k)
        # nmf = TruncatedSVD(n_components = k)
        nmf.fit(X)
        W = nmf.transform(X) #len(beers),k
        H = nmf.components_ #k,len(beers)

        all_words = []
        for group in range(k):
            #idx of the top ten words for each group
            i_words = np.argsort(H[group,:])[::-1][:top]
            words = bag[i_words]
            all_words.append(words)

            i_label = np.argsort(W[:,group])[::-1][:top]

            if print_tops:
                print('-'*10)
                print('Group:',group)
                print('WORDS')
                for word in words:
                    print('-->',word)
                print('LABELS')
                for i in i_label:
                    print('==>',labels[i])
        return W,H,nmf,tfidf,all_words

    def Plot_radial(self,entry):

        # based on the entry search, beer_list is a list of indecies to find those beers later
        if type(entry) == int:
            beer_list = np.argwhere(self.cluster_labels==entry).flatten()
        # if string search in top terms for each beer
        elif type(entry) == str and entry != '':
            beer_list = [i for i, beer in enumerate(self.beer_top_terms) if entry in beer]
            if beer_list == []:
                return None
        # if list of indecies beer_list = those indecies
        elif type(entry) == list:
            beer_list = entry
        # default is cluster 0
        else:
            return None

        plt.style.use('fivethirtyeight')
        plt.figure(figsize = (14,9))

        cluster_directions = np.linspace(0,np.pi*2,self.k+1)

        side = round(len(beer_list)**.5)
        length = float(len(beer_list))/5
        for j,i in enumerate(beer_list):
            ax = plt.subplot(side,side+1,j+1, projection='polar')
            ax.set_xticks(cluster_directions) #radial
            plt.sca(ax)

            plt.xticks(cluster_directions, ['5', '1', '2', '3', '4'], color='black', fontsize = 10)

            ax.set_yticks([]) #radius magnitude
            ax.set_title(self.beer_names[i],fontsize = 12)
            width_size = .4
            theta = cluster_directions[1:]-.5*width_size
            width = np.ones(self.k)*width_size


            radii = self.W[i]
            bars = ax.bar(theta, radii, width=width, bottom=0.0)
            # Use custom colors and opacity
            for i, bar in enumerate(bars):
                bar.set_facecolor(colors[i])
                bar.set_alpha(0.75)
        plt.tight_layout()

        return beer_list

    # ------------ 2D Plotting ------------
    def Plot_pca(self,beer_labels):

        pca = PCA(n_components=3) #3= 85.5%
        pca.fit(self.W)
        self.pca = pca
        X = pca.transform(self.W)
        y = self.cluster_labels

        # labels for lagend = top 10 words
        labels = ['{}: '.format(i+1) + ', '.join(word) for i,word in enumerate(self.top_words)]

        plt.style.use('fivethirtyeight')
        plt.figure(figsize = (14,10))

        # plots the scatter, colors = topic, size = 3rd pca feature
        for i in range(self.k):
            idx = np.argwhere(y==i)
            X_ = X[idx][:,0]

            sizes = X_[:,2]*1000+267
            sizes[sizes<50] = 50
            plt.scatter(X_[:,0],X_[:,1], s = sizes,c = colors[i],alpha = .5,label = labels[i])

        # adding beer title labels from beer_labels
        np.random.seed(None)
        max_labels = 10
        # randomly pick max_labels from beer_labels
        if beer_labels is not None:
            # incase there is less beers than max_labels
            if len(beer_labels) < max_labels:
                max_labels = len(beer_labels)

            beer_labels = np.random.choice(beer_labels,max_labels,replace = False)
        # if no labels randomly pick from all the beers
        else:
            beer_labels = np.random.choice(len(self.beer_names),10,replace = False)
        np.random.seed(42)
        # print the labels
        for i in beer_labels:
            plt.annotate(self.beer_names[i], (X[:,0][i], X[:,1][i]))

        # limit the y axis to allow for the legend to fit
        plt.ylim(np.min(X[:,1])*1.2-.04*self.k, np.max(X[:,1])*1.2)

        plt.ylabel('2nd PCA dimension (dot size is 3rd PCA dimension)')
        plt.xlabel('1st PCA dimension (dot size is 3rd PCA dimension)')

        plt.legend(loc = 'lower left')
        plt.title("PCA of New Brands (Legend shows topic's top associated terms)")

    def Plot_one_beer(self,beer_index):
        X = self.pca.transform(self.W)
        a,b = X[beer_index,0],X[beer_index,1]
        plt.scatter(a,b,color = 'gold',marker = 'D', s = 100)
        plt.annotate(self.beer_names[beer_index], (a,b))

    def Plot_one_radial(self,beer_index):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize = (4,4))

        cluster_directions = np.linspace(0,np.pi*2,self.k+1)

        ax = plt.subplot(projection='polar')
        ax.set_xticks(cluster_directions) #radial
        plt.sca(ax)

        plt.xticks(cluster_directions, ['5', '1', '2', '3', '4'], color='black', fontsize = 10)

        ax.set_yticks([]) #radius magnitude
        ax.set_title(self.beer_names[beer_index],fontsize = 12)
        width_size = .4
        theta = cluster_directions[1:]-.5*width_size
        width = np.ones(self.k)*width_size

        radii = self.W[beer_index]
        bars = ax.bar(theta, radii, width=width, bottom=0.0)
        # Use custom colors and opacity
        for i, bar in enumerate(bars):
            bar.set_facecolor(colors[i])
            bar.set_alpha(0.75)
        plt.tight_layout()


if __name__ == '__main__':
    plt.close('all')

    n_clusters = 5 # 5 recomened
    read_file = '../data/new_beer_features.pkl'

    nb = NewBeer()
    nb.Fit(read_file)
    nb.Transform(k = n_clusters)
    beers2label = nb.Plot_radial('puckering')
    nb.Plot_pca(beer_labels = beers2label)
    nb.Plot_one_beer(4)
    nb.Plot_one_radial(10)

    plt.show()
