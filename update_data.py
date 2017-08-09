import os
import pickle
import sys


from py_files.ttb_violin import TTBViolin
from py_files.sensitivity_violin import SensitivityViolin
from py_files.df_limiting import limiting
from py_files.new_beer_tool import NewBeer
from py_files.read_new_beers import ReadNewBeers
from py_files.predict_validation import PredictValidation

# check the python version
if sys.version_info[0] == 3:
    version_folder = 'data3'
else:
    version_folder = 'data'# current_folder = os.getcwd()

def update_taster_tool(read_file):
    '''
    --- Update Taster Tool ---
    reads from attvalpiv
    '''

    # ---- update sensitivites
    save_file = '{}/sensitivites.pkl'.format(version_folder)
    sen = SensitivityViolin()
    sen.Update(read_file,save_file)

    # NEED to make cleaned_df here
    # ---- update bias time plots
    # spike_read_pkl = '{}/sensitivites.pkl'.format(version_folder) # sensitivites_violin.py
    # ttb_read_csv = '{}/foco_cleaned_df.csv'.format(version_folder) # taster_spreadsheet.py
    # time_save_pkl = '{}/predict_validation.pkl'.format(version_folder)
    #
    # pv = PredictValidation()
    # pv.Make_dfs(ttb_read_csv, spike_read_pkl, save_df_path = time_save_pkl)

def update_newbeer_tool(folder):
    '''
    --- Update New Beer Tool ---
    reads from category folder
    '''

    rnb = ReadNewBeers(cold_start = True)#always cold for the time being
    file_list = rnb.Get_files_from_folders(folder)
    rnb.Read_new_beer_files(folder,from_list = True)
    text = rnb.Get_text_df()
    # rnb.Find_bad_format_files(folder)
    rnb.Make_comment_dfs()

def update_panel_tool():
    '''
    --- Update Panel Tool ---
    (under construction)
    '''
    pass


if __name__ == '__main__':
    print('-------------------------')
    print('')
    print('updating taster tool...')
    update_taster_tool(read_file = '../../media/sf_Sensory_Scientists_Only/attvalpiv.xlsx')

    print('-------------------------')
    print('')
    print('updating new beer tool...')
    update_newbeer_tool(folder = '../../media/sf_Brand_Descriptions')

    print('-------------------------')
    print('')
    update_panel_tool()
