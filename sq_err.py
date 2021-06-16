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
problems = problems.reset_index()
all_moves = pd.Series([])
for i in range(len(problems)):
    moves = []
    for m in range(len(problems.iloc[i].Moves)):
        moves.append(problems.iloc[i].Moves[m]['Description'])
    all_moves[i] = moves
problems.insert(2, "Holds", all_moves)

# %% Fixed parameters
num_holds = 11*18

# %% Changeable parameters
training_ratio = .7
num_epochs = 200
#%%
# Train test split
train = problems.sample(frac = training_ratio)
test = problems.loc[~problems.index.isin(train.index)]

# %%
def hold2vec(hold): # input hold, output one-hot vector of length num_holds
    pos = (ord(hold[0])-65) * 18 # letter A->0, ..., K-> 10
    if len(hold) == 2:
        pos += int(hold[1]) # number after the letter
    else: # two digit number
        pos += int(hold[1])*10 + int(hold[2])
    
    vec = np.zeros(num_holds)
    vec[pos-1] = 1 # arrays start at 1, so subtract 1
    return vec


def prob2vec(prob):
    vec = np.zeros(num_holds)
    for hold in prob:
        vec += hold2vec(hold)
    return vec

def grade2vec(grade): # keys.loc[train.iloc[i].Grade][1] returns integer
    vec = np.zeros(len(grades))
    vec[grade] = 1
    return vec
# %%
keys = pd.DataFrame(moon17.keys())
keys = pd.concat([keys,pd.Series(range(len(keys)))], axis = 1 )
keys.columns = [0,1]
keys = keys.set_index([0])

#%%
labels = list(range(len(moon17.keys())))
num_hidden = int((len(train) / ((num_holds + len(labels))))/2) # reference from StackExchange
# %%

eta = .01 # learning rate
var = 1 # Would want a higher learning rate for higher variance

##Initialize
W = np.random.normal(0,var,size = (32,198)) # input to hidden
V = np.random.normal(0,var,size = (1,32))[0] # hidden to output

#%%functions
def loss(y,t): # Square error
    return .5 * (y-t)**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# %%

loss_train = []
loss_test = []
for epoch in range(num_epochs):
    if epoch % 10 == 0:
        print("epoch:", epoch)
    shuffle(train)
    err = 0
    for l in range(train.shape[0]): # number of problems in training set

        # forward pass
        input_vec = prob2vec(train.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = sigmoid(np.dot(V, hidden)) 
        t = keys.loc[train.iloc[l].Grade][1]
        err += loss(16 * y, t)# multiply by 16 because we have 17 grades

        # backward pass
        EI = 16 * (16*y - t) * (1-y) * y 
        V -= eta * EI * hidden
        
        for i in range(num_hidden):
            W[i,:] -= eta * EI * (1-hidden[i]) * hidden[i] * V[i] * input_vec
    
    loss_train.append(err / train.shape[0]) # average error for that epoch

    err = 0
    for l in range(test.shape[0]):
        # forward pass
        input_vec = prob2vec(train.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = sigmoid(np.dot(V, hidden)) 
        t = keys.loc[train.iloc[l].Grade][1]
        err += loss(16 * y, t)# multiply by 16 because we have 17 grades
    loss_test.append(err / test.shape[0]) # average error for that epoch

# %%
import os
fig, ax = plt.subplots(figsize=(14,10))

fig.suptitle('Square Loss')
ax.plot(loss_train)
ax.plot(loss_test)
ax.legend(["Train", "Test"])
ax.axvline(np.argmin(loss_test), linestyle = '--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\loss_sq.png')
fig.show()
fig.savefig(image_path)

#%%
# Probabilities
correct = 0
within_1 = 0
within_2 = 0

for l in range(problems.shape[0]):
    # forward pass
    input_vec = prob2vec(problems.iloc[l].Holds)
    hidden = sigmoid(np.dot(W,input_vec))
    y = sigmoid(np.dot(V, hidden)) 

    predicted_grade = np.rint(y * 16)
    true_grade = int(keys.loc[problems.iloc[l].Grade])
    
    diff = abs(predicted_grade - true_grade)
    if diff == 0:
        correct += 1
    elif diff == 1:
        within_1 += 1
    elif diff == 2:
        within_2 += 1
within_1 += correct
within_2 += within_1
correct /= problems.shape[0]
within_1 /= problems.shape[0]
within_2 /= problems.shape[0]
accuracies = [correct, within_1, within_2]

fig, ax = plt.subplots(figsize=(12,9))
accuracies = [correct, within_1, within_2]
fig.suptitle('Square Loss Total Accuracy')
ax.bar(['Correct', 'Within 1', 'Within 2'], accuracies, width = .5)
ax.set_ylabel('Accuracy (%)')
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\accuracy_sq.png')
fig.show()
fig.savefig(image_path)

# %%
