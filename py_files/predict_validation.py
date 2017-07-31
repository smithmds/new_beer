
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
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import interp
from sklearn.preprocessing import StandardScaler
import time as Time
import sys

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()


class PredictValidation(object):
    def __init__(self):
        pass

    def Fit(self,time_file, ttb_read_csv = None, spike_read_pkl = None, save_file = None):

        if type(save_file) == str:
            print('updating data frames...')
            self.Make_dfs(ttb_read_csv, spike_read_pkl, save_df_path = save_file, taster_name = None)
        else:
            self.time = pd.read_pickle(time_file)

    def Make_dfs(self,ttb_read_csv, spike_read_pkl, save_df_path = None, taster_name = None):
        # seperating validated and not
        ttb = pd.read_csv(ttb_read_csv) # (233,263, 39) 2 NaN values
        self.ttb = ttb
        val = ttb[ttb['isValidated'] == 1] # (140,781, 39)
        notval = ttb[ttb['isValidated'] == 0] # (92,480, 39)

        val_bias = format_ttb_bias(val)
        val_bias['validated'] = 1
        notval_bias = format_ttb_bias(notval)
        notval_bias['validated'] = 0

        spike_df = pd.read_pickle(spike_read_pkl)
        self.attvalpiv = spike_df
        spike = format_spike_acc(spike_df,limiter = 'spike')
        self.spike = spike

        merge_cols = ['full_name','date','year','month']

        train = format_spike_acc(spike_df,limiter = 'train')
        cols = train.columns
        train.columns = ['train_'+col if col not in merge_cols else col for col in cols]
        self.train = train

        train_spike = pd.merge(spike,train,how = 'left',on = merge_cols)
        train_spike.fillna(0,inplace = True)

        # combining validated with spike data by merging
        yes = pd.merge(val_bias,train_spike,how = 'left', on = merge_cols)
        yes.fillna(0,inplace = True)
        # combining not-validated with spike data by merging
        no = pd.merge(notval_bias,train_spike,how = 'left', on = merge_cols)
        no.fillna(0,inplace = True)

        # combining data frames by appending not-validated to validated
        time = yes.append(no)
        time.reset_index(inplace =True)
        if type(save_df_path) == str:
            print('saving pickle file...')
            time.to_pickle(save_df_path)


        if taster_name is not None:
            # ---- what dates overlap
            yes = val_bias[val_bias['full_name']==taster_name]
            no = notval_bias[notval_bias['full_name']==taster_name]
            yes = set(yes['date'])
            no = set(no['date'])
            print(taster_name, 'overlaps on these months:', yes.intersection(no)) #there will be transition date

        self.time = time
        return time

    def Predict(self):
        y = self.time['validated']
        X = self.time[['total',
                        'bias',
                        'Majority-mcFresh',
                        'total_count',
                        'spike_acc',
                        'adj_spike_acc',
                        'train_total_count',
                        'train_spike_acc',
                        'train_adj_spike_acc']]
        y_test,y_pred,log = quick_reg(X,y)

    def Plot_bias(self,taster_name):

        # create the plot
        plt.style.use('classic')
        plt.figure(figsize = (12,6))

        # get the data ready
        time = self.time.copy()
        X = time[time['full_name']==taster_name].copy()
        now = Time.localtime()
        now_year = now.tm_year
        now_month = now.tm_mon
        now_time = now_year + (now_month-1)/12.0
        X = X[X['year']<= now_time]
        X.sort_values('date',inplace = True)

        # plot monthly average bias and major rate
        plt.plot(X['date'],X['bias']*100,label = 'Bias', color = 'midnightblue', lw = 1.3, alpha = .6)
        plt.plot(X['date'],X['Majority-mcFresh']*100,label = 'Majority Rate', color = 'purple', lw = 1.3, alpha = .6)

        # calculate year averages
        year = X.groupby('year').agg({'total_count':'sum', 'spike_acc':'mean', 'total':'sum', 'bias':'mean', 'Majority-mcFresh':'mean' })[['total_count','spike_acc','total','bias','Majority-mcFresh']]
        year.reset_index(inplace = True)

        # calculate everones average
        year_all = time.groupby('year').agg({'total_count':'sum', 'spike_acc':'mean', 'total':'sum', 'bias':'mean', 'Majority-mcFresh':'mean' })[['total_count','spike_acc','total','bias','Majority-mcFresh']]
        year_all.reset_index(inplace = True)
        year_all = year_all[year_all['year']<=now_year]
        year_all = year_all[year_all['year']>=X['year'].min()]

        plt.plot(year_all['year'],year_all['bias']*100,label = 'Mean Bias', lw = 3, color = 'gray')
        plt.plot(year_all['year'],year_all['Majority-mcFresh']*100,label = 'Mean Majority Rate', lw = 3, color = 'black')


        # finsihing stuff legened needs to be before averages
        plt.legend(loc = 'upper left')

        # plot year averages
        plt.plot(year['year'],year['bias']*100,label = 'year percent bias',lw = 3, color = 'midnightblue')
        plt.plot(year['year'],year['Majority-mcFresh']*100,label = 'year percent major', lw = 3, color = 'purple')

        # finishing the plot
        date_range = np.arange(X['year'].min(), X['year'].max()+2)
        plt.xticks(date_range,date_range-2000)
        plt.ylim(-5,30)
        plt.title(taster_name, fontsize = 16)
        plt.ylabel('Percent', fontsize = 16)
        plt.xlabel('Year (20XX)', fontsize = 16)
        plt.tight_layout()


def quick_reg(X,y):

    # NEED TO SCALE
    np.random.seed(42)
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    # mod = sm.OLS(y_train, X_train).fit()
    mod = sm.Logit(y_train, X_train).fit()

    print(mod.summary())

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X_train,y_train)
    y_pred = lr.predict_proba(X_test)
    r2 = lr.score(X_test,y_test)
    print('\n','-------------'*3)
    print('R^2 (accuracy) for X_test against y_test = ', r2)

    return y_test, y_pred, lr

def format_spike_acc(read_file, limiter = 'spike'):
    '''
    input either a pickle file name type str or a data frame
    '''

    if type(read_file) == str:
        df = pd.read_pickle(read_file)
    else:
        df = read_file

    # only spike data
    spikes = df[df['format'] == limiter].copy()

    # make time cols
    spikes['datetime'] = pd.to_datetime(spikes['date'])
    spikes['month'] = spikes['datetime'].apply(lambda x: x.month)
    spikes['year'] = spikes['datetime'].apply(lambda x: x.year)
    spikes = spikes[spikes['year']>2009] # remove bad dates

    # groupby name and time
    spikes_time = spikes.groupby(['full_name','year','month'])['correct','total_count'].sum()

    # create features
    spikes_time['spike_acc'] = spikes_time['correct']/spikes_time['total_count']
    spikes_time['adj_spike_acc'] = spikes_time['correct']/(spikes_time['total_count']+1)

    # format df
    spikes_time.reset_index(inplace = True)
    spikes_time['date'] = spikes_time['year'] + (spikes_time['month']-1)/12
    return spikes_time

def format_ttb_bias(read_file):
    '''
    input either a csv file name type str or a data frame
    '''

    if type(read_file) == str:
        df = pd.read_csv(read_file)
    else:
        df = read_file

    # only production beer
    bias = df[df['TestType']=='pr'].copy()

    # make time cols
    bias['datetime'] = pd.to_datetime(bias['SessionDate'])
    bias['month'] = bias['datetime'].apply(lambda x: x.month)
    bias['year'] = bias['datetime'].apply(lambda x: x.year)
    bias = bias[bias['year']>2009] # remove bad dates


    bias['total'] = 1 # to sum in groupby (not needed but...)

    # groupby time
    bias_time = bias.groupby(['RegName','year','month']).agg({'mcFresh': 'sum', 'total':'sum','Majority-mcFresh':'mean'})

    # create features
    bias_time['bias'] = bias_time['mcFresh']/bias_time['total']

    # format df
    bias_time.reset_index(inplace = True)
    bias_time['date'] = bias_time['year'] + (bias_time['month']-1)/12
    cols = bias_time.columns.tolist()
    cols[0] = 'full_name'
    bias_time.columns = cols

    return bias_time

if __name__ == '__main__':

    plt.close('all')
    spike_read_pkl = '../{}/sensitivites.pkl'.format(version_folder) # sensitivites_violin.py
    ttb_read_csv = '../{}/foco_cleaned_df.csv'.format(version_folder) # taster_spreadsheet.py
    time_read_pkl = '../{}/predict_validation.pkl'.format(version_folder)

    a = format_ttb_bias(ttb_read_csv)


    pv = PredictValidation()
    pv.Fit(time_read_pkl, ttb_read_csv, spike_read_pkl, save_file = None)
    # pv.Predict()

    # pv.Plot_bias('billy bletcher')
    pv.Plot_bias('soren daugaard')
    # pv.Plot_bias('dana sedin')
    # pv.Plot_bias('lindsay barr')
    # pv.Plot_bias('marie kirkpatrick')
    # pv.Plot_bias('jeff biegert')
    plt.show()
