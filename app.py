import matplotlib
matplotlib.use('Agg')

'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
uses javascript code from html5 up
https://html5up.net/
July 2017
'''
import os
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_from_directory
import pickle
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import time
import sys

# from StringIO import StringIO
from py_files.ttb_violin import TTBViolin
from py_files.sensitivity_violin import SensitivityViolin
from py_files.df_limiting import limiting
from py_files.new_beer_tool import NewBeer
from py_files.read_new_beers import ReadNewBeers
from py_files.predict_validation import PredictValidation

app = Flask(__name__)
title_ = 'Seeing Taste'
brand_options_ = 'ft | ra | ftc | daw | cbw'
test_options_ = 'pr | ip | sls | s | t'
current_folder = os.getcwd()
# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()


# We'd normally include configuration settings in this call

@app.route('/seeing-taste')
def seeing_taste():
    return render_template('index.html',
        title = title_)

@app.route('/update-all')
def update_all():
    return render_template('update-all.html',
        title = title_)

@app.route('/new-beer', methods=['GET'])
def new_beer():
    read_file = '{}/comment_text_df.pkl'.format(version_folder)

    nb = NewBeer()
    nb.Fit(read_file)
    beer_names_lst = ['Beer Name']+list(nb.beer_names)
    return render_template('new-beer.html',
            title = title_,
            beer_names = beer_names_lst)

@app.route('/new-beer-plots', methods=['POST'])
def new_beer_plots():


        # initialize items
        # -----------------------------

    n_clusters = 5 # 5 recomened
    read_file = '{}/comment_text_df.pkl'.format(version_folder)
    save_folder ='static/images/plots'

    # delete any plot image that exists in save_folder
    for file_name in os.listdir(save_folder):
        os.remove('{}/{}'.format(save_folder,file_name))

    # start the NewBeer Class
    nb = NewBeer()
    nb.Fit(read_file)
    nb.Transform(k = n_clusters)

    # the list for the drop down menu
    beer_names_lst = ['Beer Name']+list(nb.beer_names)

        # request input items
        # -----------------------------

    # get the search term (has priority over the radio buttons)
    beer_search = str(request.form['beer_word']).lower()
    try:
        # can't get a request if it doesnt exist
        radio_buttons = str(request.form['option'])
    except:
        radio_buttons = ''

    # request the compare/highlight beer option
    highlight_beer = [str(request.form['highlight']).lower(),
                    str(request.form['highlight2']).lower()]

    name_search = str(request.form['beer_name_search']).lower()
        # identify which beers are going to be plotted
        # -----------------------------

    # if both beers for the compare is selected
    if highlight_beer[0] != 'beer name' and highlight_beer[1] != 'beer name':
        beer_search = [
            np.argwhere(nb.beer_names == highlight_beer[0])[0][0],
            np.argwhere(nb.beer_names == highlight_beer[1])[0][0]]

        beer_search_input = 'Compare ' + highlight_beer[0] + ' and ' + highlight_beer[1]
    # if both are blank
    elif beer_search == '' and radio_buttons == '' and name_search == '':
        beer_search = None
        beer_search_input = 'None'

    # if the text field is blank
    elif beer_search == '' and name_search == '':
        beer_search = int(radio_buttons)
        beer_search_input = 'Topic {}'.format(beer_search + 1)

    # if search by name
    elif name_search != '':
        # 1- i because of 'beer name' in first list position
        beer_search = [i-1 for i, beer in enumerate(beer_names_lst) if name_search in beer]
        beer_search_input = 'Name = ' + name_search
    # try to find the closest word
    elif beer_search not in nb.bag_of_words:
        choices = nb.bag_of_words
        beer_search = process.extractOne(beer_search, choices)[0]
        beer_search_input = 'Beer term = ' + beer_search
    else:
        beer_search_input = 'Beer term = ' + beer_search


        # Plotting
        # -----------------------------

    # plot and save the radial plot
    beers2label = nb.Plot_radial(beer_search)
    save_radial = '{}/radial_plot_{}.png'.format(save_folder,int(time.time()))
    plt.savefig(save_radial, facecolor = 'white')

    # plot and save the pca
    nb.Plot_pca(beer_labels = beers2label)

    # plot the highlited beers on the pca
    top_beer_terms = []
    for beer in highlight_beer:
        if beer == 'beer name':
            top_beer_terms.append('')
        else:
            highlight_idx = np.argwhere(nb.beer_names == beer)[0][0]
            nb.Plot_one_beer(highlight_idx)
            # get the associated words for the comparison
            top_terms_ = list(nb.beer_top_terms[highlight_idx][:15])
            top_beer_terms.append([beer] + top_terms_)

    save_pca = '{}/pca_plot_{}.png'.format(save_folder,int(time.time()))
    plt.savefig(save_pca, facecolor = 'white')


        # Finishing items
        # -----------------------------

    radial_plt_lst = []


    return render_template('new-beer-plot.html',
            title = title_,
            pca_fig = save_pca,
            radial_fig = save_radial,
            search_term = beer_search_input,
            beer_names = beer_names_lst,
            one_radial_lst = radial_plt_lst,
            top_beer1_words = top_beer_terms[0],
            top_beer2_words = top_beer_terms[1])

@app.route('/new-beer/updating', methods=['GET'])
def updating():

    # folder = 'files/doc_files' #file to read the .doc new beers from
    folder = '~\oldcherry\shared\QA\Fort Collins\Sensory\P50s not in Sample Manager\Pilot Brewery'
    rnb = ReadNewBeers(cold_start = False)
    files, beers = rnb.Read_new_beer_files(folder)
    text, names = rnb.Get_text_df()
    # rnb.Find_bad_format_files(folder)
    rnb.Make_comment_dfs()

    return '''
                <html>
                	<head>
                		<title>Updated</title>
                        <h1>Updating Complete</h1>
                    </head>
                </html>
            '''

@app.route('/export', methods=['GET'])
def export():
    return render_template('export.html',
            title = title_,
            brand_options = brand_options_,
            test_options = test_options_)

@app.route('/export/file',methods = ['POST'])
def exporting():
    save_file_name = '{}/inter_files/test_csv.csv'.format(version_folder)
    brewery = str(request.form['option1']).lower()
    validated = str(request.form['option2']).lower()

    brand = [str(request.form['brand1']).lower(), str(request.form['brand2']).lower()]
    if brand[0] == '' and brand[1] == '':
        brand[0] = 'all'
    test_type = [str(request.form['test1']).lower(), str(request.form['test2']).lower()]
    if test_type[0] == '' and test_type[1] == '':
        test_type[0] = 'all'

    date_range = [str(request.form['option3']).lower(), str(request.form['date1']).lower()]

    limits = [brewery,validated,brand,test_type,date_range]

    df = limiting(version_folder,limits)
    df.index.rename('OriginalIndex',inplace = True)
    # there is a second original index (unknown col) here from the first .csv save
    df.to_csv(save_file_name)
    return send_from_directory(directory = '{}/inter_files'.format(version_folder),
            filename='test_csv.csv')

@app.route('/generic', methods=['GET'])
def generic():
    return render_template('generic.html',
            title = title_)

@app.route('/elements', methods=['GET'])
def elements():
    return render_template('elements.html',
            title = title_)

@app.route('/taster', methods=['GET'])
def taster():
    read_file = '{}/sensitivites.pkl'.format(version_folder)
    sen = SensitivityViolin()
    sen.Fit(read_file)

    # to create the taster drop down
    taster_name_lst = list(sen.df['full_name'].unique())
    taster_name_lst.sort()
    taster_name_lst = ['Taster Name Drop Down']+taster_name_lst
    return render_template('taster.html',
            title = title_,
            taster_names = taster_name_lst)

@app.route('/taster-plots', methods=['POST'] )
def trust():
    save_folder = 'static/images/plots'

    # delete any plot image that exists in save_folder
    for file_name in os.listdir(save_folder):
        os.remove('{}/{}'.format(save_folder,file_name))

    # request the taster names from the form
    name = str(request.form['name'])
    drop_name = str(request.form['taster_dropdown1'])
    if name == '' and drop_name != 'Taster Name Drop Down':
        name = drop_name

    name2 = str(request.form['name2'])
    drop_name2 = str(request.form['taster_dropdown2'])
    if name2 == '' and drop_name2 != 'Taster Name Drop Down':
        name2 = drop_name2

    if name2 != '':
        names = [name,name2]
    else:
        names = [name]

    # ---- plot the ttb violin
    read_file = '{}/trustworthiness_ratings.pkl'.format(version_folder)
    ttb = TTBViolin(read_file)
    choices = ttb.df_ttb.index.tolist()
    names_out = [process.extractOne(name, choices)[0] for name in names]

    ttb.plot_tasters(names_out)
    save_ttb = '{}/ttb_violin_{}.png'.format(save_folder,int(time.time()))
    plt.savefig(save_ttb)

    # ---- sense plots
    top = 10
    update_file = 'files/attvalpiv.csv'
    read_file = '{}/sensitivites.pkl'.format(version_folder)
    sen = SensitivityViolin()
    # sen.Update(update_file, save_path = read_file)
    sen.Fit(read_file)
    choices = sen.groups['full_name'].unique().tolist() #each have their own choices
    name_out = process.extractOne(names[0], choices)[0]

    sen.Plot_violin(name_out, 'percent_adj',top = top, na_thresh = 0)
    save_sense = '{}/sense_violin_{}.png'.format(save_folder,int(time.time()))
    plt.savefig(save_sense)

    # ---- bias time plot
    read_file = '{}/predict_validation.pkl'.format(version_folder)
    pv = PredictValidation()
    pv.Fit(read_file,
            ttb_read_csv = None, # these three are for updating
            spike_read_pkl = None,
            save_file = None)
    pv.Plot_bias(name_out)
    save_bias = '{}/bias_plot_{}.png'.format(save_folder,int(time.time()))
    plt.savefig(save_bias, facecolor = 'white')


    # to create the taster drop down
    taster_name_lst = list(sen.df['full_name'].unique())
    taster_name_lst.sort()
    taster_name_lst = ['Taster Name Drop Down']+taster_name_lst



    return render_template('taster-plots.html',
            title = title_,
            ttb_fig = save_ttb,
            sense_fig = save_sense,
            bias_fig = save_bias,
            taster_names = taster_name_lst)

@app.route('/taster/updating', methods=['GET'])
def taster_updating():

    # ---- update sensitivites
    # read_file = 'files/attvalpiv.csv'
    read_file = '~/S:/QA/Fort Collins/Sensory/P50s not in Sample Manager/Sensory Scientists Only/attvalpiv.xlsx'
    save_file = '{}/sensitivites.pkl'.format(version_folder)
    sen = SensitivityViolin()
    sen.Update(read_file,save_file)

    # # ---- update bias time plots
    # spike_read_pkl = '{}/sensitivites.pkl'.format(version_folder) # sensitivites_violin.py
    # ttb_read_csv = '{}/foco_cleaned_df.csv'.format(version_folder) # taster_spreadsheet.py
    # time_save_pkl = '{}/predict_validation.pkl'.format(version_folder)
    #
    # pv = PredictValidation()
    # pv.Make_dfs(ttb_read_csv, spike_read_pkl, save_df_path = time_save_pkl)

    return '''
                <html>
                	<head>
                		<title>Updated</title>
                        <h1>Updating Complete</h1>
                    </head>
                </html>
            '''

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8155, debug=True)
