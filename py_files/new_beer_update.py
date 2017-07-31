'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''

import spacy
import pandas as pd
import pickle
from py_files.read_docs import read_new_beer_files, find_bad_format_files
from py_files.new_beer_features import make_top_words
import sys

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()

# current_folder = os.getcwd()

def update_all_files(folder,cold_start = False):

    if cold_start:
        print('This is a COLD START.')

    # from read_docs
    print('Reading in new .doc files...')
    comments, files, beers = read_new_beer_files(folder, cold_start = cold_start)
    bad_files = find_bad_format_files(folder)

    # from new_beer_features
    drop_lst = ['aroma', 'note', 'notes', 'sl', 'slight', 'light', 'hint', 'bit', 'little', 'lot', 'touch', 'character', 'some', 'something', 'retro', 'thing', ' ']

    print('\nLoading Spacy English...')
    nlp = spacy.load('en')
    feats, all_text = make_top_words(drop_lst,nlp)
    feats.columns = ['bitter_sweet_ratio','avg_color','rbg_code', 'aroma_counter']
    feats['all_text'] = all_text

    # saving the files
    feats.to_pickle('{}/new_beer_features.pkl'.format(version_folder))
    pickle.dump(bad_files, open('{}/bad_files.pkl'.format(version_folder), "wb" ))
    print('Finished saving all files')

    return feats, bad_files

if __name__ == '__main__':
    # run from main folder as run py_files/new_beer_update
    folder = 'files/doc_files' #file to read the .doc new beers from

    feats, bad_files = update_all_files(folder,cold_start = True)
