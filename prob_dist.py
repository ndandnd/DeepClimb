#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
#%%
with open('moon17.json') as f:
    data = json.load(f)

# Preprocessing
problems = pd.DataFrame(data)
problems = problems.drop(["DateInserted", 'Method', 'MoonBoardConfigurationId', 'Setter',
       'Rating', 'Repeats', 'Holdsetup', 'IsBenchmark', 'IsAssessmentProblem'])
problems = problems.T


grades = list(problems.Grade.unique())
grades.sort()
moon17 = {elem : pd.DataFrame for elem in grades}

for key in moon17.keys():
    moon17[key] = problems[:][problems.Grade == key].reset_index(drop=True)
    moon17[key].set_index('Name', inplace=True)
#%%
num_probs = []
for grade in grades:
    num_probs.append(moon17[grade].shape[0])
plt.bar(grades, num_probs)
# %%
with open('moon16.json') as f:
    data = json.load(f)

# Preprocessing
problems = pd.DataFrame(data)
problems = problems.drop(["DateInserted", 'Method', 'MoonBoardConfigurationId', 'Setter',
       'Rating', 'Repeats', 'Holdsetup', 'IsBenchmark', 'IsAssessmentProblem'])
problems = problems.T


grades = list(problems.Grade.unique())
grades.sort()
moon16 = {elem : pd.DataFrame for elem in grades}

for key in moon16.keys():
    moon16[key] = problems[:][problems.Grade == key].reset_index(drop=True)
    moon16[key].set_index('Name', inplace=True)
#%%
num_probs = []
for grade in grades:
    num_probs.append(moon16[grade].shape[0])
plt.bar(grades, num_probs)
# %%
