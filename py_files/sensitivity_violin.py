import matplotlib
matplotlib.use('Agg')

'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import re
import sys

current_folder = os.getcwd()

name_dict = {'alonzo chunn':'alonzo chun',
                'bethany beert':'bethany beers',
                'bill bletcher':'billy bletcher',
                'brandon ingram':'brandon ingraham',
                'casey kjolhede':'casey kjohede',
                'jason schawb':'jason schwab',
                'jonathon walker':'jonathan walker',
                'kaitlyn peot':'kaitlyn poet',
                'kaityln peot':'kaitlyn poet',
                'lisa ammann':'lisa amman',
                'marie kirkpartrick':'marie kirkpatrick',
                'mark fishcer':'mark fischer',
                'mark fishcher':'mark fischer',
                'mrak fischer':'mark fischer',
                'michael shaw':'mike shaw',
                'name not listed':'unknown',
                'nate kacmarek':'nate kaczmarek',
                'nathan kaczmarek':'nate kaczmarek',
                'nick ampe':'nick ample',
                'penny gill-stuart':'penny gillstuart',
                'select your name.':'unknown',
                'travis van eron':'travis von eron',
                'zac white':'zach white'}

def in_name_dict(x):
    '''
    checks if the name is in the name dict and returns the correct name if true else returns orignal name
    '''
    if x in name_dict.keys():
        return name_dict[x]
    else:
        return x

class SensitivityViolin(object):

    def __init__(self):
        pass

    def Update(self,read_file,save_path = None):
        df = pd.read_csv(read_file, encoding = 'latin1')
        df.columns = [re.sub(' ','_',col.lower()) for col in df.columns.tolist()]

        # cleaning format column
        df['format'] = df['format'].apply(lambda x: re.sub('ip ', '', x.lower()))
        df['format'] = df['format'].apply(lambda x: re.sub('spikes', 'spike', x.strip()))
        df['format'] = df['format'].apply(lambda x: re.sub('line ', '', x))
        df['format'] = df['format'].apply(lambda x: re.sub('night ', '', x))
        df['format'] = df['format'].apply(lambda x: re.sub('  ', '_', x))
        df['test'] = [0 if 'train' in form else 1 for form in df['format'].values]

        # cleaning threshold grouping into 3 groups
        df['threshold_level'] = df['threshold_level'].apply(lambda x: str(x).lower().strip())
        d_level = {'3x':2, '6x':3, '1.5x':1, '4.5x':2, '9x':4, '25%':2, 'ip beer':5,
            'na':5, '15%':1, '20ppb':2, '20%':2, '75%':4, '40 ppb':4,
            '12x':4, '6x%':1, '1x':1, 'nan':5, '4.5':2, '30%':3}
        df['level_group'] = df['threshold_level'].apply(lambda x: d_level[x])

        # cleaning the other columns
        df['full_name'] = df['full_name'].apply(lambda x: x.lower().strip())
        df['flavor'] = df['flavor'].apply(lambda x: re.sub(' ','_',x.lower().strip()))
        df['flavor'] = df['flavor'].apply(lambda x: re.sub('butryic', 'butyric', x.lower().strip()))

        df.drop(['unnamed:_12','unnamed:_13'], axis = 1, inplace = True)

        df['full_name'] = df['full_name'].apply(in_name_dict)

        # df['timestamp'] = pd.to_datetime(df['date']) #timestamp type
        #.day , .month, .year, .to_julian_date()
        self.df = df
        if type(save_path) == str:
            print('updated pickle file', save_path)
            df.to_pickle(save_path)


    def Group_flavors(self,test_format = 'spike'):
        self.test_format = test_format
        # --- creates the features
        df = self.df[self.df['format']==test_format]
        groups = df.groupby(['flavor','full_name']).sum()
        groups = groups[['correct','incorrect','test','level_group']]
        groups['total'] = groups['correct']+groups['incorrect']
        groups['percent'] = groups['correct']/groups['total']
        # groups['test_percent'] = groups['test']/groups['total']
        # groups['avg_level'] = 1.0*groups['level_group']/groups['total']
        groups['percent_adj'] = (groups['correct'])/(groups['total']+1) # +1 helps adjust in favor of more tastings over less
        groups.reset_index(inplace = True)
        groups.drop(['incorrect','test','level_group'],axis = 1,inplace = True)
        groups.dropna(inplace = True)

        self.groups = groups

    def Fit(self,read_file):
        self.df = pd.read_pickle(read_file)
        self.Group_flavors(test_format = 'spike')

    def Plot_violin(self,taster_name, measure_col, top = 10, na_thresh = 0):

        plt.style.use('classic')
        # ------ create pivot matricies
        scores = self.groups.pivot(index = 'full_name', columns = 'flavor', values = measure_col)#percent_adj
        totals = self.groups.pivot(index = 'full_name', columns = 'flavor', values = 'total')
        correct = self.groups.pivot(index = 'full_name', columns = 'flavor', values = 'correct')

        scores = scores.dropna(axis = 1, thresh = na_thresh) # drops cols with fewer than thresh non-null values
        totals = totals.dropna(axis = 1, thresh = na_thresh)
        correct = correct.dropna(axis = 1, thresh = na_thresh)

        self.scores = scores
        self.totals = totals
        self.correct = correct

        score_plot = np.array([scores[col].dropna().values for col in scores.columns.tolist()])
        vio_labels = np.array(scores.columns.tolist())

        dots = scores.loc[taster_name].fillna(-1).values # tasters individual scores
        taster_totals = totals.loc[taster_name]
        taster_correct = correct.loc[taster_name]

        idots = dots.argsort()[::-1] # indicies for the tasters individual scores
        taster_totals = taster_totals[idots]
        taster_correct = taster_correct[idots]

        # ------ create figure
        plt.figure(figsize = (17,8.3))
        violin_parts = plt.violinplot(score_plot[idots][:top], showmeans=True, vert = True)
        # sets the axis and labels
        plt.xticks(np.arange(1,top+1),vio_labels[idots][:top],rotation = 30, fontsize = 16)


        # ---- change the color of the violin if the taster got them all right
        for i,pc in enumerate(violin_parts['bodies']):
            if taster_correct[i]/taster_totals[i] == 1:
                pc.set_facecolor('midnightblue')
            else: pc.set_facecolor('steelblue')
            pc.set_edgecolor('black')
        violin_parts['cbars'].set_color('purple')
        violin_parts['cbars'].set_alpha(.5)

        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')

        # ---- plot the tastes totals as the marker at the location of their ability
        # create text markers string
        markers = [r"$ {} $".format(int(total)) for total in taster_totals.fillna(0)]

        plt.scatter(1,dots[idots][0],marker = 'o', s = 500, c = 'white', alpha = 1,label = taster_name)
        for i in np.arange(0,top):
            plt.scatter(i+1,dots[idots][i],marker = 'o', s = 500, c = 'white', alpha = 1)
            plt.scatter(i+1,dots[idots][i],marker = markers[i], s = 200, c = 'midnightblue')

        # Set y limits
        plt.ylim(-.03,1.1)
        plt.title("{} Data Violin Plots of Top {}/{} Sensitivites (red line = mean)".format(self.test_format, top, len(vio_labels)),fontsize = 20)
        plt.ylabel('Sensitivity Score ({})'.format(measure_col),fontsize = 20)
        plt.legend(loc = 'upper right')
        plt.tight_layout()


        # save the figure

        # create output DataFrame
        taster_info = pd.DataFrame(taster_totals)
        taster_info.columns =['{}-total'.format(taster_name)]
        num_correct = correct.loc[taster_name]
        taster_info['{}-correct'.format(taster_name)] = num_correct[idots]
        taster_info['{}-score'.format(taster_name)] = dots[idots]

        self.taster_info = taster_info


if __name__ == '__main__':
    plt.close('all')


    if sys.version_info[0] == 3:
        save_file = '../data3/sensitivites.pkl'
    else:
        save_file = '../data/sensitivites.pkl'


    sen = SensitivityViolin()
    # sen.Update('../files/attvalpiv.csv', save_path = save_file)
    sen.Fit(save_file)
    taster_name = 'soren daugaard' # soren_daugaard is very good score of .75
    sen.Plot_violin(taster_name, 'percent_adj',top = 10, na_thresh = 10)
    plt.show()
