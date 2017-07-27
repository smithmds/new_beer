'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''

import spacy
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
import re
import string
import py_files.webcolors #https://webcolors.readthedocs.io/en/1.7/
import sys

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()


def clean(raw_str_desc, nlp, slang_lst = None):
    '''
    raw_str_desc = type list of type unicode strings
    slang_lst = not yet added
    '''
    # 1--- lower case the text
    lower_str_desc = [text.lower() for text in raw_str_desc]

    # 2--- remove punctuation and replace it with a space
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    no_punc_str_desc = [regex.sub(' ', text) for text in lower_str_desc]

    # 3--- correct long vowel sounds
    # doesnt get words like soo with exaclty 2 vowels that are doubled
    # it does tag words like raspberry with 3 'r' but doesnt change them to anything
    words_desc = [text.split() for text in no_punc_str_desc]
    for i, words in enumerate(words_desc):
        for j, word in enumerate(words):
            c = Counter(word)
            #below: (1) = 1 return, [0]enter into string, [1] = second value the number of times
            if c.most_common(1)[0][1] >= 3: #identify a word with 3 or more letters
                letter = c.most_common(1)[0][0]
                str_to_replace = letter*c.most_common(1)[0][1]

                replacement_str_2 = letter*2 #replace with two letters
                new_word_2 = re.sub(str_to_replace,replacement_str_2,word)
                replacement_str_1 = letter #replace with one letter
                new_word_1 = re.sub(str_to_replace,replacement_str_1,word)

                # checks the pobability of 2 or 1 repeated letters
                if nlp(new_word_2)[0].prob > nlp(new_word_1)[0].prob:
                    new_word = new_word_2
                else:
                    new_word = new_word_1

                # print( words_desc[i][j], '-->', new_word)
                words_desc[i][j] = new_word

    # adds and and to the end so it never tries to find nouns a adj out of range
    str_desc = [' '.join(text) + ' and and' for text in words_desc]

    # 4--- create the spacy representation of the text
    docs = [nlp(text) for text in str_desc]

    # returns a list of docs
    return docs

# ----------- color functions

def closest_colour(requested_colour):
    '''
    returns the closest color name to a rgb number
    '''
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    '''
    checks the actual and closest name
    calls requested_colour
    '''
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def color_math(word_lst):
    '''
    averages colors based on words in list input
    calls get_colour_name
    '''
    rgb_lst = []
    for word in word_lst:
        try:
            if word != u'white': #this made everything so much better
                rgb_lst.append(np.array(webcolors.name_to_rgb(word)))
        except:
            pass
    if rgb_lst == []:
        closest_name = 'unknown'
        rgb_avg = 'unknown'
    else:
        rgb_avg = tuple(np.mean(np.array(rgb_lst),axis = 0))
        rgb_avg = tuple([int(c) for c in rgb_avg])
        actual_name, closest_name = get_colour_name(rgb_avg)
        rgb_avg = tuple([int(c) for c in rgb_avg])

    # print( "Actual colour name:", actual_name, ", closest colour name:", closest_name)
    return closest_name, rgb_avg

def get_aroma_words(raw_str_desc,nlp,drop_lst):
    '''
    returns all nouns and noun bigrams from aroma as a counter.
    processes the words too slightly - impovements can be made here
    calls clean
    '''
    docs = clean(raw_str_desc,nlp)
    all_text = [' '.join([word.string.strip() for word in doc]) for doc in docs]
    aroma = docs[1]


    aroma_nouns = []
    for word in aroma:
        if word.pos_ == u'NOUN':
            word = word.lemma_.strip()

            # make this better
            word = re.sub('malty','malt',word)
            word = re.sub('fruity','fruit',word)
            if word not in drop_lst:
                aroma_nouns.append(word)

    aroma_biagrams = [aroma_nouns[i-1] + ' ' + aroma_nouns[i] for i in range(1,len(aroma_nouns))]
    counter = Counter(aroma_nouns+aroma_biagrams)
    return counter, all_text

def make_top_words(drop_lst, nlp):
    '''
    returns a feature matrix of [bitter_sweet_ratio, color_name, rgb, aroma_counts],[all_text]
    calls get_aroma_words, color_math
    inputs nlp = spacy.load('en')
    '''
    all_text = []
    feats = []
    df = pd.read_pickle('../{}/inter_files/beer_batches.pkl'.format(version_folder))
    beer_names = df.index.tolist()
    for beer_num in range(0,len(df)):
        aroma_counts, clean_comments = get_aroma_words(df.values[beer_num],nlp,drop_lst)

        # bitter sweet ratio
        c = Counter(clean_comments[2].split()) #for flavor only
        # could add magnitude scaling to this ratio
        if c['sweet']>0:
            bitter_sweet_ratio = round(float(c['bitter'])/c['sweet'],3)
        else:
            bitter_sweet_ratio = 5

        # get average color
        # does better for lighter and red colors than browns
        # might have something to do with the foam may want to do two seperate
        word_desc_visual = clean_comments[0].split() #visual only
        color_name, rgb = color_math(word_desc_visual)

        feats.append([bitter_sweet_ratio,color_name,rgb,aroma_counts])
        all_text.append(clean_comments) # all text in one big string

    feats = pd.DataFrame(feats)
    feats.index = beer_names

    return feats, all_text


if __name__ == '__main__':
    # run -i v1_desc
    if not 'nlp' in locals():
        print("Loading English Module...")
        nlp = spacy.load('en')

    drop_lst = ['aroma','note','notes','sl','slight','light','hint','bit','little', 'lot','touch','character','some','something','retro','thing',' ']

    feats, all_text = make_top_words(drop_lst,nlp)
    # have a metric to see how well the tasters agree
    c = feats[3][0]
    not_one = [k for k,v in c.iteritems() if v>1]

    print('try $ feats or $ all_text')
