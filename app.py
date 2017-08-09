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


''' --- TASTER TOOL ---'''

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


''' --- NEW BEER TOOL ---'''

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

        # narrow down the categories
        # -----------------------------
    categories = ['tests_and_trials',
                 'in_process',
                 'finished_product',
                 'market_beers',
                 'wood_cellar',
                 'cider']

    cat_inputs = [] # list

    cat_boxes = ['cat' + str(i+1) for i in range(6)] # ['cat1', etc]

    for i, cat in enumerate(cat_boxes):
        if request.form.get(cat):
        	cat_inputs.append(categories[i]) #list of categories

    cat_idx = []
    for cat_input in cat_inputs:
        cat_idx.extend([i for i, cat in enumerate(nb.category) if cat_input in cat])

    full_cat_index = list(set(cat_idx))
        # request input items
        # -----------------------------

    one_beer = str(request.form['one_beer']).lower()
    two_beer = str(request.form['two_beer']).lower()
    beer_search_term = str(request.form['beer_search']).lower()

    beer1_idx = None
    beer2_idx = None

    # if one beer is selected == closest 5
    if one_beer != 'beer name' and  two_beer == 'beer name':
        beer1_idx = np.argwhere(nb.beer_names == one_beer)[0][0]
        dist, idxs = nb.Find_neighbors(beer1_idx)

        # restrict based on what category
        full_cat_index.append(beer1_idx)# add to beer categories
        idxs = [i for i in idxs if i in full_cat_index]

        beer_search = list(idxs[:6])
        beer_search_input = 'Similar to: '+one_beer

    # if both beers for the compare is selected
    elif one_beer != 'beer name' and two_beer != 'beer name':
        beer1_idx = np.argwhere(nb.beer_names == one_beer)[0][0]
        beer2_idx = np.argwhere(nb.beer_names == two_beer)[0][0]
        full_cat_index.append(beer1_idx)# add to beer categories
        full_cat_index.append(beer2_idx)# add to beer categories

        beer_search = [beer1_idx, beer2_idx]
        beer_search_input = 'Comparing {} and {}'.format(one_beer, two_beer)

    # if the beer search term is not blank
    elif beer_search_term != '':
        name_or_term = str(request.form['search_option'])

        # if we are searching names
        if name_or_term == 'name_entry':
            beer_search = [i-1 for i, beer in enumerate(beer_names_lst) if beer_search_term in beer]
            beer_search_input = 'Name = ' + beer_search_term

        # if we are searching terms
        elif name_or_term == 'term_entry':
            if beer_search_term in nb.bag_of_words:
                beer_search = [i for i, beer in enumerate(nb.beer_top_terms) if beer_search_term in beer]
            else: #if that term doesnt exist
                beer_search = np.random.choice(range(len(nb.beer_names)), size = 1, replace = False)
                beer_search_term = 'None'
            beer_search_input = 'Beer term = ' + beer_search_term

    else:
        try: # to request the cluster radio buttons
            radio_buttons = int(str(request.form['option']))
            beer_search = np.argwhere(nb.cluster_labels==radio_buttons).flatten()
            beer_search_input = 'Cluster {}'.format(radio_buttons+1)
        except: # no entry, select all
            beer_search = range(len(nb.beer_names))
            beer_search_input = 'None'


    # add these to the full_cat_index
    # if beer2_idx is not None and beer2_idx not in full_cat_index:
    #     full_cat_index.append(beer2_idx)
    # if beer1_idx is not None and beer1_idx not in full_cat_index:
    #     full_cat_index.append(beer1_idx)

    # limit to those of the selected categories
    beer_search = [i for i in beer_search if i in full_cat_index]

    # if its too many, restrict the number to 30
    if len(beer_search) > 30:
        beer_search =np.random.choice(range(len(nb.beer_names)), size = 30, replace = False)
        beer_search_input = beer_search_input + '(Limit 30)'


        # Plotting
        # -----------------------------

    # plot and save the radial plot
    beers2label = nb.Plot_radial(beer_search)
    save_radial = '{}/radial_plot_{}.png'.format(save_folder,int(time.time()))
    plt.savefig(save_radial, facecolor = 'white')

    # plot and save the pca
    nb.Plot_pca(beer_labels = beers2label,limited = full_cat_index)

    # plot the highlited beers on the pca
    highlight_beer = [one_beer,two_beer]
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



    return render_template('new-beer-plot.html',
            title = title_,
            pca_fig = save_pca,
            radial_fig = save_radial,
            search_term = beer_search_input,
            beer_names = beer_names_lst,
            top_beer1_words = top_beer_terms[0],
            top_beer2_words = top_beer_terms[1],
            categories_chosen = ' | '.join(cat_inputs))


''' --- EXPORT TOOL ---'''

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

''' --- UPDATING ---'''

@app.route('/update-all')
def update_all():
    return render_template('update-all.html',
        title = title_)

@app.route('/taster/updating', methods=['GET'])
def taster_updating():

    # ---- update sensitivites
    # read_file = 'files/attvalpiv.csv'
    read_file = '../../media/sf_Sensory_Scientists_Only/attvalpiv.xlsx'
    save_file = '{}/sensitivites.pkl'.format(version_folder)
    sen = SensitivityViolin()
    sen.Update(read_file,save_file)

    # ---- update bias time plots
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

@app.route('/new-beer/updating', methods=['GET'])
def updating():

    # folder = 'files/doc_files' #file to read the .doc new beers from
    folder = '../../media/sf_Brand_Descriptions'
    rnb = ReadNewBeers(cold_start = True)#always cold for the time being
    file_list = rnb.Get_files_from_folders(folder)
    rnb.Read_new_beer_files(folder,from_list = True)
    text = rnb.Get_text_df()
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

''' --- GENERIC ---'''

@app.route('/generic', methods=['GET'])
def generic():
    return render_template('generic.html',
            title = title_)

@app.route('/elements', methods=['GET'])
def elements():
    return render_template('elements.html',
            title = title_)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8155, debug=True)
