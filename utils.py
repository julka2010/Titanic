import pandas as pd
import numpy as np

sex_to_num = {'male': 0, 'female': 1}
num_to_sex = ['male', 'female']
dock_to_num = {'': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
num_to_dock = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
title_to_num = {'Jonkheer': 4, 'Dr': 1, 'Miss': 2, 'Mrs': 3, 'the Countess': 3, 'Capt': 0, 'Don': 0, 'Dona': 3, 'Mme': 4, 'Major': 0, 'Sir': 0, 'Lady': 3, 'Rev': 4, 'Ms': 2, 'Col': 4, 'Mlle': 2, 'Master': 0, 'Mr': 4}

def read_and_clean_datafile(filepath):
    df = pd.read_csv(filepath)
    df.loc[df.loc[:, 'Sex'] == 'male', 'Sex'] = sex_to_num['male']
    df.loc[df.loc[:, 'Sex'] == 'female', 'Sex'] = sex_to_num['female']
    df.loc[:, 'known_age'] = pd.Series(df.Age.notna().astype(int))
    df['Title'] = pd.Series(
        np.array([t.split(',')[1].split('.')[0].strip() for t in df.Name]),
        index=df.index,
    )
    for i, t in df.iterrows():
        df.loc[i, 'Title'] = title_to_num[t.Title]
    return df
