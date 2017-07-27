'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import sys

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()

# import os

# current_folder = os.getcwd()
ttb_cols = ['mcClarity', 'mcAroma', 'mcFlavor', 'mcMouthfeel', 'mcFresh']

def clean_mc_cols(x):
    if x > 0:
        return x-1
    elif x == -1 or x == 0:
        return 0
    else:
        return 0

word_cols = ['RegName','Package','TestType','Flavor']

def clean_word_cols(x):
    # add punctuation drops?
    if type(x) == str or type(x) == unicode:
        return x.lower()
    else:
        return u'unknown'

comment_cols = ['cClarity', 'cAroma', 'cFlavor', 'cMouthfeel', 'cFresh', 'cNotFresh']

#ashville dict
name_dict = {'stacia':'stacia janes',
                'dana':'dana sedin',
                'lb':'lindsay barr',
                'lindsay':'lindsay barr',
                'lindsay barrrrrrr':'lindsay barr',
                'lindsay b':'lindsay barr',
                'pbouckaert':'peter bouckaert',
                'stacia':'stacia janes',
                'valerie':'valerie patenotte',
                'j':'unknown',
                'your favorite person':'unkown',
                'andy sturm':'andrew sturm', # start foco
                'lindsay guerdrum':'lindsay barr',
                'matty gilliland':'matt gilliland',
                'michelle hobrook':'michelle holbrook',
                'pat murfin':'patrick murfin',
                'phil pollick':'philip pollick',
                'alex':'unknown',
                'alison shultz':'alis schultz',
                'alyse':'unknown',
                'andrea dimatto':'andrea dimatteo',
                'andy m':'andy mitchell',
                'anja':'unknown',
                'becki':'unknown',
                'bh':'unknown',
                'billy b':'billy bletcher',
                'bill bletcher':'billy bletcher',
                'bla *':'unknown',
                'brent':'brent radke',
                'brian':'unknown',
                'carrie':'unknown',
                'chelsea':'chelsea lawler',
                'chris m':'chris mccombs',
                'cof':'unknown',
                'corey':'cory hudson',
                'dawn':'unknown',
                'doug':'unknown',
                'drew b':'drew bombard',
                'drew bombasrd':'drew bombard',
                'emily':'emily dufficy',
                'emma dufficy':'emily dufficy',
                'eng':'unknown',
                'erin':'erin hughes',
                'gabriel':'unknown',
                'hr':'unknown',
                'isabelle':'unknown',
                'jan':'unknown',
                'jenna':'unknown',
                'jesse':'unknown',
                'josh':'unknown',
                'kayla':'unknown',
                'kaylyn':'kaylyn kirkpatrick',
                'larry':'larry shriver',
                'lc':'unknown',
                'luke':'unknown',
                'matty':'unknown',
                'matty3':'unknown',
                'mclean':'unknown',
                'pad':'unknown',
                'pat':'patrick murfin',
                'pkg':'unknown',
                'pur':'unknown',
                'qu':'unknown',
                'ross koeings':'ross koenigs',
                'ross koenig':'ross koenigs',
                'sandi':'unknown',
                'spike':'unknown',
                'steve':'unknown',
                'sven':'unknown',
                'tamar':'tamar banner',
                'thomas':'unknown',
                'tim horjes':'tim horejs',
                'truj':'unknown',
                'tyler f':'tyler foos',
                'walt':'unknown',
                'wayne':'unknown',
                'zach b':'zach baitinger'
                }

def in_name_dict(x):
    '''
    checks if the name is in the name dict and returns the correct name if true else returns orignal name
    '''
    if x in name_dict.keys():
        return unicode(name_dict[x])
    else:
        return unicode(x)

def clean_testtype(x):
    if 'pr' in x:
        return u'pr'
    elif x == 's' or x == 'spike':
        return u's'
    elif 'ip' in x:
        return u'ip'
    elif 'sls' in x:
        return u'sls'
    elif x =='t':
        return u't'
    else:
        return u'unknown'



def clean_taster_df(df):
    '''
    cleans the word and ttb columns
    NEED TO ADD comment cols here
    '''
    # drops none for ash and 27 for foco
    df.dropna(how = 'all',axis = 0,inplace = True)

    for col in word_cols:
        df[col] = df[col].apply(clean_word_cols)
    df['RegName'] = df['RegName'].apply(in_name_dict)
    df['TestType'] = df['TestType'].apply(clean_testtype)

    for col in ttb_cols:
        df[col] = df[col].apply(clean_mc_cols)

    return df

def add_cols_before_groupby(df):
    '''
    creates panel size, number not TTB, and majoirty cols
    '''
    # add panel size col
    col_name = 'PanelSize'
    num_in_tasting = df.groupby('BrewNumber')['p50'].count()
    num_in_tasting.rename(col_name,inplace = True)
    df = df.join(num_in_tasting,on='BrewNumber')
    df[col_name].fillna(0,inplace = True)

    # add number of not ttb for each col
    for col in ttb_cols:
        num_not_ttb = df.groupby('BrewNumber')[col].sum()
        not_ttb_name = 'NumNotTTB-{}'.format(col)
        num_not_ttb.rename(not_ttb_name,inplace = True)
        df = df.join(num_not_ttb, on = 'BrewNumber')
        df['Majority-{}'.format(col)] = 1.0*df[col]*df[not_ttb_name]/df[col_name]

    return df


def clean_and_save_tasting_csv():

    print('loading ashville')
    file_name = 'CS P50 FTC and AVL - full table data.xlsx'
    foco_raw_og = pd.read_excel(file_name,sheetname = 0)
    print('loading fort collins')
    file_name = 'CS P50 FTC and AVL - full table data.xlsx'
    ash_raw_og = pd.read_excel(file_name,sheetname = 1)

    print('cleaning ashville')
    ash = clean_taster_df(ash_raw_og.copy())
    ash = add_cols_before_groupby(ash)

    ash.to_csv('../{}/ash_cleaned_df.csv'.format(version_folder))

    print('cleaning fort collins')
    foco = clean_taster_df(foco_raw_og.copy())
    foco = add_cols_before_groupby(foco)

    foco.to_csv('../{}/foco_cleaned_df.csv'.format(version_folder))
    return ash_raw_og, foco_raw_og

def search_data(limits, df):
    '''
    based on limits returns a smaller dataframe
    limits = type tuple (col,compare,term)
    col = column
    compare = <,>,=
    term = what to limit by
    '''
    for limit in limits:
        col = limit[0]
        compare = limit[1]
        term = limit[2]
        if compare == '=':
            df = df[df[col] == term]
        elif compare == '>':
            df = df[df[col] > term]
        elif compare == '<':
            df = df[df[col] < term]
        else:
            pass

    return df

def group_tasters(df):
    tasters = df.groupby('RegName').agg({
        'p50' : ['count'],
        'mcClarity' : ['mean'],
        'mcAroma' : ['mean'],
        'mcFlavor' : ['mean'],
        'mcMouthfeel' : ['mean'],
        'mcFresh' : ['mean'],
        'Majority-mcClarity' : ['mean'],
        'Majority-mcAroma' : ['mean'],
        'Majority-mcFlavor' : ['mean'],
        'Majority-mcMouthfeel' : ['mean'],
        'Majority-mcFresh' : ['mean']
        })
    tasters.columns = ['AromaBias','ClarityBias','BodyBias','TasteBias',
                        'Experience','OverallBias',
                        'AromaMajority','ClarityMajority','BodyMajority','TasteMajority','OverallMajority'
                        ]
    return tasters


if __name__ == '__main__':

    # ash_raw, foco_raw = clean_and_save_tasting_csv()

    ash_ = pd.read_csv('../{}/ash_cleaned_df.csv'.format(version_folder))
    foco_ = pd.read_csv('../{}/foco_cleaned_df.csv'.format(version_folder))

    limits = [('isValidated','=',0),
                ('Flavor','=','ft'),
                ('TestType','=','pr')]
    limits = [('Flavor','=','ft'),
                ('TestType','=','pr')]
                # ('Package', '=', 'bottle')]

    ash = search_data(limits,ash_)
    ash = group_tasters(ash)

    foco = search_data(limits,foco_)
    foco = group_tasters(foco)



    # difference in tasters from ash and foco

    col = 'Aroma' # Clarity Body Overall Taste
    col2plot = '{}Bias'.format(col)
    with plt.xkcd():
        plt.figure(figsize = (7,8))
        ash_vio = plt.violinplot(ash[col2plot])
        ash_vio['bodies'][0].set_color('blue')
        plt.scatter([1],[0],color = 'blue',label = 'Ash')
        foco_vio = plt.violinplot(foco[col2plot])
        foco_vio['bodies'][0].set_color('purple')
        plt.scatter([1],[0],color = 'purple',label = 'Foco')
        plt.ylim(-.01,1.01)
        plt.ylabel('Taster Bias',fontsize = 18)
        plt.title('Violin Plot of {} for Validated PR FT'.format(col),fontsize = 18)
        plt.xticks([])
        plt.legend()
        # plt.savefig('files/foco_vs_ash_aroma_bias.png')
    # plt.show()
