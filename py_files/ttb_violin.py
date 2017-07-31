
'''
class produces a violin plot with tasters for the TTB trustworthiness ratings
requires 'data/trustworthiness_ratings.pkl'

Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# current_folder = os.getcwd()
import sys

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()



class TTBViolin(object):
    def __init__(self,read_file):
        # self.df_ttb = pd.read_pickle(current_folder+'/data/trustworthiness_ratings.pkl')
        self.df_ttb = pd.read_pickle(read_file)

        plt.style.use('classic')
        data = self.df_ttb.values
        pos = [0,1,2,3,4]
        f = plt.figure(figsize = (10,6))
        f.set_facecolor('lightgray')


        violin_parts = plt.violinplot(data, pos, widths=0.7, showmeans=True)
        plt.ylim(.9, 4.5)
        plt.xlim(-.5,4.5)
        plt.yticks([1,2,3,4],['1','2','3','4'],fontsize = 16)
        plt.xticks([0,1,2,3,4],['Average','Taste','Clarity','Aroma','Body'],fontsize=16)
        plt.ylabel('Trust Score (4 = most trustworthy)', fontsize = 16)
        plt.title('Trustworthiness Distributions (red line = mean)', fontsize = 16)

        for i,pc in enumerate(violin_parts['bodies']):
            if i == 0:
                pc.set_facecolor('midnightblue')
            else: pc.set_facecolor('steelblue')
            pc.set_edgecolor('black')
        violin_parts['cbars'].set_color('purple')
        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')


    def plot_tasters(self, name_lst, limit = 2):
        # plt.style.use('ggplot')
        colors = ['purple','midnightblue']
        if type(name_lst) == str:
            name_lst = [name_lst]

        scores = [self.df_ttb.loc[name].values for name in name_lst[:limit]]

        for i,name,score in zip(np.arange(len(colors)),name_lst,scores):
            plt.plot([0,1,2,3,4],score, '--', ms = 12, marker = 'D',label = name, c = colors[i], linewidth = 2, alpha = 1)
        plt.legend(loc='upper center', bbox_to_anchor=(.5, 1),
              ncol=3, fancybox=True, shadow=False,scatterpoints = 1)

    def save_plot(self,file_path):
        plt.savefig(file_path)

if __name__ == '__main__':

    plt.close('all')
    name = 'ben barrett'
    names = ['eric unger','patrick murfin']
    vio = TTBViolin('../{}/trustworthiness_ratings.pkl'.format(version_folder))
    vio.plot_tasters(names)
    # vio.save_plot('figures/ttb_violin.png')
    print('class = vio.')
    plt.show()
