

from scipy.misc import comb
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import sys
import time

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()

def my_round(float_num,round_to = 3):
    string_num = str(float_num).split('.')
    rounded_str = string_num[0] + '.' + string_num[1][:round_to]
    return float(rounded_str)

def P_binomial(n,k,p):
    '''
    n = number of times (panels size)
    k = how many we are looking at (number of panelists)
    p = probability of action (bias P(!TTB))
    '''
    return comb(n,k) * p**k * (1-p)**(n-k)

def Binomial_dist(prob_lst):
    '''
    calls P_binomial
    '''
    mean_bias = np.mean(prob_lst)
    panel_size = len(prob_lst)
    return [P_binomial(panel_size,k,mean_bias) for k in range(0,panel_size+1)]

def Get_time(n_months):
    '''
    if n_months = 6 then it returns the date time 6 months before the current month
    '''
    month = time.localtime().tm_mon - n_months
    year = time.localtime().tm_year
    return year + (month-1)/12

class BinomialProbability(object):

    def __init__(self,alpha = .9725, months = 6):
        if alpha >= 1: # if they put in as percent
            alpha = alpha/100.0
        self.alpha = alpha
        self.months = months # number of months back to score bias


    def Get_data(self,read_folder):
        time_file = read_folder + '/predict_validation.pkl'
        bytime = pd.read_pickle(time_file)
        self.bytime = bytime
        last_months = bytime[bytime['date']>=Get_time(self.months)]
        self.biasies = last_months.groupby('full_name')['bias'].mean()

        foco_file = read_folder + '/foco_cleaned_df.csv'
        foco = pd.read_csv(foco_file)
        self.foco = foco

    def P_one(self,brew_number):
        '''
        for a brew number returns the actual num !TTB, the min number of !TTB needed, and the probability (confidence) that it isnt chance all in a list
        '''
        panel = self.foco[self.foco['BrewNumber'] == brew_number].copy()
        self.panel = panel
        validated = panel[panel['isValidated'] == 1]
        val_tasters = validated['RegName'].values

        # drop na removes people who havent tasted in the last so many months
        bias_lst = self.biasies[val_tasters].dropna()
        self.bias_lst = bias_lst

        bi_dist = Binomial_dist(bias_lst)
        p_not_chance = np.cumsum(bi_dist)
        min_n_not_ttb = np.argwhere(p_not_chance > self.alpha).flatten()[0]


        n_not_ttb = int(validated['mcFresh'].sum())
        return [n_not_ttb,
                min_n_not_ttb,
                my_round(p_not_chance[min_n_not_ttb],3),
                len(val_tasters),
                my_round(bias_lst.mean())]

    def Get_p_all(self):
        brew_numbers = self.foco['BrewNumber'].unique()
        # p_all = [self.P_one(brew_num) for brew_num in brew_numbers[last:]]

        p_all = []
        for brew_num in brew_numbers:
            try:
                p_all.append(self.P_one(brew_num))
            except:
                p_all.append([np.NAN,np.NAN,np.NAN,np.NAN,np.NAN])

        pdf = pd.DataFrame(p_all)
        pdf.columns = ['n_!ttb','min_n','min_confidence','panel_size', 'mean_bias']
        pdf.index = brew_numbers
        return pdf



if __name__ == '__main__':

    # bias = foco[['RegName','isValidated','BrewNumber','PanelSize','NumNotTTB-mcFresh']]

    marie = .06
    billy = .09
    lindsay = .3
    dana = .025
    soren = .15

    a = [marie,billy,dana,soren]

    bp = BinomialProbability(99, months = 12)
    bp.Get_data('../{}'.format(version_folder))
    p = bp.P_one(160109081)
    al = bp.Get_p_all()

    # removing bad brew numbers
    brew_number_limits = [int(9e7)]
    brew_number_limits.append(int(str(time.localtime().tm_year - 2000) + str(time.localtime().tm_mon/100).split('.')[1] + str(time.localtime().tm_mday/100).split('.')[1])*1000)
    al.reset_index(inplace = True)
    al = al[al['index'] >= brew_number_limits[0]]
    al = al[al['index'] <= brew_number_limits[1]]
    al.sort_values('index',inplace = True)

    bad = al[al['n_!ttb']>=al['min_n']]
