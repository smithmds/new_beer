'''
Code written and developed by Jan Van Zeghbroeck
https://github.com/janvanzeghbroeck
'''
import pandas as pd
import sys


def limiting(folder, limiting_list):

    print(limiting_list)
    # 0 Brewery
    if limiting_list[0] == 'foco':
        df = pd.read_csv('{}/foco_cleaned_df.csv'.format(folder))
    elif limiting_list[0] == 'ash':
        df = pd.read_csv('{}/ash_cleaned_df.csv'.format(folder))
    else:
        foco = pd.read_csv('{}/foco_cleaned_df.csv'.format(folder))
        foco['Brewery'] = 'fort collins'
        ash = pd.read_csv('{}/ash_cleaned_df.csv'.format(folder))
        ash['Brewery'] = 'ashville'
        df = pd.concat([foco,ash])

    # print('brewery',df.shape)

    # 1 Validated
    if limiting_list[1] == 'yes':
        df = df[df['isValidated'] == 1]
    if limiting_list[1] == 'no':
        df = df[df['isValidated'] == 0]
    else:
        pass

    # print('validated',df.shape)

    # 2 Brand
    if limiting_list[2][0] == 'all':
        pass
    else:
        df1 = df[df['Flavor'] == limiting_list[2][0]]
        df2 = df[df['Flavor'] == limiting_list[2][1]]
        df = pd.concat([df1,df2])

    # print('brand', df.shape)

    # 3 Test Type
    if limiting_list[3][0] == 'all':
        pass
    else:
        df1 = df[df['TestType'] == limiting_list[3][0]]
        df2 = df[df['TestType'] == limiting_list[3][1]]
        df = pd.concat([df1,df2])

    # print('test type', df.shape)

    # 4 Date Range
    if limiting_list[4][0] == 'all':
        pass
    else:
        # convert to pandas date time type
        input_date = pd.to_datetime(limiting_list[4][1])
        df['SessionDate'] = pd.to_datetime(df['SessionDate'])

        if limiting_list[4][0] == 'less':
            df = df[df['SessionDate'] < input_date]
        elif limiting_list[4][0] == 'greater':
            df = df[df['SessionDate'] > input_date]
        else:
            pass

    # print('date range', df.shape)

    return df


if __name__ == '__main__':
    # check the python version
    if sys.version_info[0] == 3:
        folder = '../data3'
    else:
        folder = '../data'
    # brewery, validated, [brand 1,2], [test 1,2], [date sign,date]
    limits = ['both','yes',['ft',''],['all',''],['greater','06/10/16']]

    df = limiting(folder,limits)
