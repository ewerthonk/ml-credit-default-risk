# -*- coding: utf-8 -*-

from random import sample
import pandas as pd
from pathlib import Path
from IPython.display import display, Markdown

project_directory = Path(__file__).resolve().parents[2]
raw_data_directory = project_directory / 'data' / 'raw'

def list_datasets(folder='raw'):
    """
"""
    from pathlib import Path
    from IPython.display import display

    data_directory = Path(__file__).resolve().parents[2] / 'data' / folder
    datasets_in_folder = tuple([path.name for path in data_directory.rglob('*.csv')])
    display(datasets_in_folder)


def overview_data(csv_file_name, 
                  display_mode='all',
                  display_option='expanded',
                  sample_mode='head',
                  sample_size=10):
    """"
"""  
    display_option_validation = ('expanded', 'condensed')
    sample_mode_validation = ('head', 'random')
    
    if display_option not in display_option_validation:
        raise ValueError(f'display_option must be one of {display_option_validation}.')
    if sample_mode not in  sample_mode_validation:
        raise ValueError(f'sample_mode must be one of {sample_mode_validation}.')

    path_to_csv_file = Path(raw_data_directory / csv_file_name)
    df = pd.read_csv(path_to_csv_file)
    pd.reset_option('display.max_columns')

    if (display_option=='expanded') & (len(df.columns) > pd.options.display.max_columns):
        pd.set_option('display.max_columns', len(df.columns))

    display(Markdown('#### Dataset name (.csv)'))
    print(path_to_csv_file.name)

    if (display_mode=='sample') or (display_mode=='all') or ('sample' in display_mode):
        if sample_mode == 'head':
            display(Markdown('#### Dataset Records'),
                    df.head(sample_size))
        else:
            display(Markdown('#### Dataset Random Samples'),
                   df.sample(sample_size, random_state=42))

    if (display_mode=='info') or (display_mode=='all') or ('info' in display_mode):
        display(Markdown('#### Dataset info'))
        df.info()

    if (display_mode=='describe') or (display_mode=='all') or ('describe' in display_mode):
        display(Markdown('#### Dataset descriptive statistics'),
                df.describe(include='all'))

    pd.reset_option('display.max_columns')


def describe_features(dataset, display_option='expanded'):
    """
"""
    display_option_validation = ('expanded', 'condensed')
    if display_option not in display_option_validation:
        raise ValueError(f'display_option must be one of {display_option_validation}.')

    path_to_csv_file = Path(raw_data_directory / 'HomeCredit_columns_description.csv')
    df = pd.read_csv(path_to_csv_file, index_col=0)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns = {'Table': 'Dataset', 'Row': 'Column'}, inplace=True)
    pd.reset_option('display.max_rows')

    if dataset in ['application_train.csv', 'application_test_student.csv']:
        df_description_table = df[(df['Dataset']=='application_{train|test}.csv')]
    else:
        df_description_table = df[(df['Dataset']==dataset)]

    if display_option == 'expanded':
        pd.set_option('display.max_rows', len(df_description_table))
        max_description_length = df_description_table['Description'].str.len().max()
        max_special_length = df_description_table['Special'].str.len().max()
        if max_special_length > pd.options.display.max_colwidth:
            with pd.option_context('display.max_colwidth', max_special_length+1):
                display(df_description_table)                
        elif max_description_length > pd.options.display.max_colwidth:
            with pd.option_context('display.max_colwidth', max_description_length+1):
                display(df_description_table)
    else:
        display(df_description_table)

    pd.reset_option('display.max_rows')


def describe_feature(dataset, feature_or_column, display_option='expanded'):
    """
"""
    display_option_validation = ('expanded', 'condensed')
    if display_option not in display_option_validation:
        raise ValueError(f'display_option must be one of {display_option_validation}.')
    
    path_to_csv_file = Path(raw_data_directory / 'HomeCredit_columns_description.csv')
    df = pd.read_csv(path_to_csv_file, index_col=0)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns = {'Table': 'Dataset', 'Row': 'Column'}, inplace=True)

    if dataset in ['application_train.csv', 'application_test_student.csv']:
        df_description_instance = df[(df['Dataset']=='application_{train|test}.csv')
                                     & (df['Column']==feature_or_column)]
    else:
        df_description_instance = df[(df['Dataset']==dataset)
                                     & (df['Column']==feature_or_column)]

    if display_option == 'expanded':
        description_length = df_description_instance['Description'].str.len()
        special_length = df_description_instance['Special'].str.len()
    if description_length > pd.options.display.max_colwidth:
        with pd.option_context('display.max_colwidth', description_length):
            display(df_description_instance)
    elif special_length > pd.options.display.max_colwidth:
        with pd.option_context('display.max_colwidth', special_length):
            display(df_description_instance)
    else:
        display(df_description_instance)
    

def create_dataframe(csv_file_name):
    """
"""
    path_to_csv_file = Path(raw_data_directory / csv_file_name)
    df = pd.read_csv(path_to_csv_file)

    return df


