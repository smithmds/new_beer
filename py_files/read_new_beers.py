'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''


import docx
import os
import pandas as pd
import pickle
import re
import sys
import numpy as np

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()

def process_table(table):
    n_rows = len(table.rows) #19
    # n_cols = len(table.columns)#6

    # creates a table
    table_ = []
    for i in range(n_rows):
        row = table.rows[i]
        row_ = []
        for cell in row.cells:
            row_.append(cell.text)
        table_.append(row_)

    # creates data frame
    df = pd.DataFrame(table_)
    df.columns = df.iloc[0,:].values #rename columns
    df.drop(0,axis = 0,inplace = True) #delete column name row
    return df

class ReadNewBeers(object):

    def __init__(self,cold_start = False):
        self.cold_start = cold_start
        self.wanted_comments = ['Visual Comment  ', 'Aroma Comment  ', 'Mouthfeel Comments  ']


    def Update_inter_file(self, file_path, data):
        if not self.cold_start:
            pickle_file = list(pickle.load(open(file_path,'rb')))
            data = pickle_file + data

        pickle.dump( data, open(file_path, "wb" ) )
        return data

    def Read_new_beer_files(self,folder):

        '''
        reads all the tables from docx files and puts it into a list of lists of datafrems
        beers(file) --> beer --> tables --> table
        '''

        # file_name = 'Cherry Brown FTR Sep 16 15.docx'
        file_names = []
        beers = []

        if self.cold_start:
            finished_files = []
            print('!!! COLD START !!!')
        else:# load what files we have alredy looked at
            finished_files = list( pickle.load( open('{}/inter_files/file_names.pkl'.format(version_folder),'rb')))

        for file_name in os.listdir(folder):
            tables = []
            # if file is a .docx file and hasnt been seen yet
            if file_name.split('.')[-1] == 'docx' and file_name not in finished_files:

                print('checking', file_name)
                file_str = '{}/{}'.format(folder,file_name)
                doc = docx.Document(file_str)

                # append the name to the first location if it exists
                if len(doc.tables) > 1: # are there at least 2 tables
                    # are there at least 2 rows and 3 columns
                    if len(doc.tables[1].rows) > 1 and len(doc.tables[1].columns) > 2:
                        tables.append(doc.tables[1].rows[1].cells[2].text)

                # for each table in list of tables
                for idx, table in enumerate(doc.tables):
                    # if table is long enough
                    if len(table.rows)>2 and len(table.columns)>1:
                        # if it has a wanted comment field
                        if table.rows[2].cells[1].text in self.wanted_comments:
                            # get the text from the table 2 after
                            table.rows[2].cells[1].text
                            df = process_table(doc.tables[idx+2])
                            tables.append(df)
            # check if it got read in correctly
            # name and three dfs and all tables same length
            if len(tables) == 4 and len(tables[1]) == len(tables[2]) and len(tables[1]) == len(tables[3]):
                beers.append(tables)
                file_names.append(file_name)
                print('--> saved', file_name)

        # either update the files or save them fresh (cold start)
        file_path_df = '{}/inter_files/docx_to_dfs.pkl'.format(version_folder)
        self.beer_tables = self.Update_inter_file(file_path_df,beers)

        file_path_df = '{}/inter_files/file_names.pkl'.format(version_folder)
        self.file_names = self.Update_inter_file(file_path_df,file_names)

        return self.file_names, self.beer_tables

# assume beer[0] = name str
# assume beer[1:] = data frams of same shape
    def Get_text_df(self):
        '''
        turns the raw table data into a data frame of comments
        '''
        text_names = [word.split()[0].lower() for word in self.wanted_comments]
        beer_names = []
        beer_text = []
        for beer in self.beer_tables:
            beer_names.append(beer[0])# the zero entry is the beer's name
            df_ = pd.DataFrame()
            for i, df in enumerate(beer[1:]):
                df_[text_names[i]] = df.iloc[:,-1]
            beer_text.append(df_)
        self.beer_text_dfs = beer_text
        self.beer_names = beer_names
        return self.beer_text_dfs, self.beer_names

    def Make_comment_dfs(self):
        '''
        makes a dataframe with 2 cols: beer_name and comment_text (all comments as a string)
        '''
        text = [' '.join([' '.join(col) for col in df.values.T]) for df in self.beer_text_dfs]
        text_df = pd.DataFrame(list(zip(self.beer_names,text)))
        text_df.columns = ['beer_name','comment_text']
        # lower case the beer names
        text_df['beer_name'] = text_df['beer_name'].apply(lambda x: x.lower())
        # add underscores to beer names
        text_df['beer_name'] = text_df['beer_name'].apply(lambda x: re.sub(' ', '_', x))

        # combine duplicates
        text_df = text_df.groupby('beer_name').aggregate(lambda x: ' '.join(list(x)))

        text_df.reset_index(inplace = True)

        # save / update the comment_text_df.pkl file
        text_df.to_pickle('{}/comment_text_df.pkl'.format(version_folder))
        self.text_df = text_df


    def Find_bad_format_files(self):
        '''
        Returns the files that werent uploaded
        '''
        good_files = list( pickle.load( open('{}/inter_files/file_names.pkl'.format(version_folder),'rb')))
        all_files = os.listdir(folder)
        self.bad_files = [file_name for file_name in all_files if file_name not in good_files]
        return self.bad_files



if __name__ == '__main__':

    folder = 'files/doc_files' # .doc new beer files
    rnb = ReadNewBeers(cold_start = True)
    files, beers = rnb.Read_new_beer_files(folder)
    text, names = rnb.Get_text_df()
    rnb.Find_bad_format_files()
    rnb.Make_comment_dfs()
