# -*- coding: utf-8 -*-

def overview_data(csv_file_name: str):
    """"
"""
    import pandas as pd
    from pathlib import Path
    from IPython.display import display, Markdown

    raw_data_directory = Path(__file__).resolve().parents[2].joinpath('data', 'raw')

    path_to_csv_file = Path(raw_data_directory.joinpath(csv_file_name))
    df = pd.read_csv(path_to_csv_file)

    if len(df.columns) > pd.options.display.max_columns:
        threshold = len(df.columns)//2
        with pd.option_context('display.max_columns', threshold+1):   
            display(Markdown('### Dataset name (.csv)'),
                    Markdown('---'))
            print(path_to_csv_file.name)
            display(Markdown('### 5 samples from the data (with all columns)'),
                    Markdown('---'),
                    df.sample(5, random_state=42).iloc[:,:threshold],
                    df.sample(5, random_state=42).iloc[:,threshold:],
                    Markdown('### Dataframe info'),
                    Markdown('---'))
            df.info()
            display(Markdown('### Dataframe descriptive statistics'),
                    Markdown('---'),
                    df.iloc[:,:threshold].describe(include='all'),
                    df.iloc[:,threshold:].describe(include='all'))
    else:
        display(Markdown('### Dataset name (.csv)'),
                Markdown('---'))
        print(path_to_csv_file.name)
        display(Markdown('### 5 samples from the data'),
                Markdown('---'),
                df.sample(5, random_state=42),
                Markdown('### Dataframe info'),
                Markdown('---'))
        df.info()
        display(Markdown('### Dataframe descriptive statistics'),
                Markdown('---'),
                df.describe(include='all'))