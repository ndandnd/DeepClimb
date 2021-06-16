#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import os
#%%
with open('moon17.json') as f:
    data = json.load(f)

# Preprocessing
problems = pd.DataFrame(data)
problems = problems.drop(["DateInserted", 'MoonBoardConfigurationId', 'Setter',
       'IsAssessmentProblem', "Rating", "Holdsetup"])
problems = problems.T


grades = list(problems.Grade.unique())
grades.sort()
moon17 = {elem : pd.DataFrame for elem in grades}

for key in moon17.keys():
    moon17[key] = problems[:][problems.Grade == key].reset_index(drop=True)
    moon17[key].set_index('Name', inplace=True)

problems = problems[problems['Repeats']>0] # only ones that are repeated


# only consider problems 6A+ to 7A+ (7 categories)
problems = moon17['6A+'].append([moon17['6B'], moon17['6B+'], moon17['6C'], moon17['6C+'], moon17['7A'], moon17['7A+']])


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
training_ratio = .8

#validation_ratio = .5 # percent of remaining that will be used for validation
num_epochs = 10
#%%
# Train test split
train = problems.sample(frac = training_ratio)
test = problems.loc[~problems.index.isin(train.index)]
#test_valid = problems.loc[~problems.index.isin(train.index)]
#valid = test_valid.sample(frac = validation_ratio)
#test = test_valid.loc[~test_valid.index.isin(valid.index)]

# %%
keys = pd.DataFrame(moon17.keys())
keys = pd.concat([keys,pd.Series(range(len(keys)))], axis = 1 )
keys.columns = [0,1]
keys = keys.set_index([0])

#%%
labels = list(range(len(problems['Grade'].unique())))
num_hidden = int((len(train) / ((num_holds + len(labels))))/2) # reference from StackExchange

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

## changed grade2vec to len(labels) instead of len(labels)
def grade2vec(grade): # keys.loc[train.iloc[i].Grade][1] returns integer
    vec = np.zeros(len(labels))
    vec[grade] = 1
    return vec
# %%

eta = .01 # learning rate
# We want to vary our learning rate
# scale by int(np.log2(2 + repeats))
#benchmark_scale = 3 # Because anyone can submit a problem and give it any grade they like there’s a feature in the app that let’s you see only problems that are considered 'Benchmarks'

var = 1 # Would want a higher learning rate for higher variance
##Initialize
W = np.random.normal(0,var,size = (num_hidden,198)) # input to hidden
V = np.random.normal(0,var,size = (len(labels),num_hidden)) # hidden to output

#%%functions
def loss(y,t): # cross entropy error (easy to calculate in classification problem)
    epsilon = 1e-100
    return -np.log(np.dot(y,t) + epsilon) # to prevent division by 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def softmax(vec):
    max = np.max(vec) # To reduce computation
    exp = np.exp(vec - max) # by reducing exp size
    return exp / np.sum(exp)


# %%

loss_train = []
loss_test = []
for epoch in range(num_epochs):
    if epoch % 3 == 0:
        print("epoch:", epoch)
    shuffle(train)
    err = 0
    for l in range(train.shape[0]): # number of problems in training set
        
        # scale the learning for different quality of problem
        learn_scale = 1 #int(np.log(train.iloc[l].Repeats + 2)) # multiplies to learning rate
        # if train.iloc[0].IsBenchmark:
        #     learn_scale *= benchmark_scale

        # forward pass
        input_vec = prob2vec(train.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = softmax(np.dot(V, hidden))
        t = grade2vec(keys.loc[train.iloc[l].Grade][1] - 2) # subtract 2 b/c start at 6A+ 
        err += loss(y,t)

        # backward pass
        for j in range(len(labels)): # update V
            V[j,:] -= learn_scale * eta * (y[j] - t[j]) * hidden 
        
        for i in range(num_hidden):
            W[i,:] -= learn_scale * eta * np.dot(y-t, V[:,i]) * hidden[i] * (1-hidden[i]) * input_vec # 198 dim vector
        # for k in range(num_holds): # update W
        #     W[:,k] -= eta * np.dot(y-t, V[:,i]) * hidden * (1-hidden)
    loss_train.append(err / train.shape[0]) # average error for that epoch

    err = 0
    for l in range(test.shape[0]):
        # forward pass
        input_vec = prob2vec(test.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = softmax(np.dot(V, hidden))
        t = grade2vec(keys.loc[test.iloc[l].Grade][1] - 2) # subtract 2 b/c start at 6A+
        err += loss(y,t)
    loss_test.append(err / test.shape[0]) # average error for that epoch
# %%
# pd.DataFrame(W).to_csv("tuned_matrices/ce2017W_cleaned.csv", header = None, index = None)
# pd.DataFrame(V).to_csv("tuned_matrices/ce2017V_cleaned.csv", header = None, index = None)

# %%

fig, ax = plt.subplots(figsize=(14,10))

fig.suptitle('Cross Entropy Loss')
ax.plot(loss_train)
ax.plot(loss_test)
ax.legend(["Train", "Test"])
ax.axvline(np.argmin(loss_test), linestyle = '--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
#image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\loss_CE_cleaned.png')
fig.show()
#fig.savefig(image_path)

#%%
# Probabilities
correct = 0
within_1 = 0
within_2 = 0

for l in range(problems.shape[0]):
    # forward pass
    input_vec = prob2vec(problems.iloc[l].Holds)
    hidden = sigmoid(np.dot(W,input_vec))
    y = softmax(np.dot(V, hidden))

    predicted_grade = np.argmax(y)
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
fig.suptitle('Cross Entropy Total Accuracy')
ax.bar(['Correct', 'Within 1', 'Within 2'], accuracies, width = .5)
ax.set_ylabel('Accuracy (%)')
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\accuracy.png')
fig.show()
#fig.savefig(image_path)
#%%
def disperse1(t, rate): # takes in the true one-hot vector, disperses it to places above and below it by the rate
    peak = np.argmax(t)
    if peak == 0:
        t[peak] = rate + (1-rate)/2
        t[peak+1] = (1-rate)/2
    elif peak == len(t)-1:
        t[peak] = rate + (1-rate)/2
        t[peak-1] = (1-rate)/2
    else:
        t[peak] = rate
        t[peak-1] = (1-rate)/2
        t[peak+1] = (1-rate)/2
    return

# %% 
# Method 2 (dispersion)
# Initialize

dispersion_rate = .6 # rate of dispersion
W = np.random.normal(0,var,size = (num_hidden,198)) # input to hidden
V = np.random.normal(0,var,size = (len(labels),num_hidden)) # hidden to output

loss_train = []
loss_test = []
for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print("epoch:", epoch)
    shuffle(train)
    err = 0
    for l in range(train.shape[0]): # number of problems in training set

        # forward pass
        input_vec = prob2vec(train.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = softmax(np.dot(V, hidden))
        t = grade2vec(keys.loc[train.iloc[l].Grade][1])
        disperse1(t, dispersion_rate)
        err += loss(y,t)

        # backward pass
        for j in range(len(labels)): # update V
            V[j,:] -= eta * (y[j] - t[j]) * hidden 
        
        for i in range(num_hidden):
            W[i,:] -= eta * np.dot(y-t, V[:,i]) * hidden[i] * (1-hidden[i]) * input_vec # 198 dim vector
    loss_train.append(err / train.shape[0]) # average error for that epoch

    err = 0
    for l in range(test.shape[0]):
        # forward pass
        input_vec = prob2vec(test.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = softmax(np.dot(V, hidden))
        t = grade2vec(keys.loc[test.iloc[l].Grade][1])
        #disperse1(t, dispersion_rate) # When testing, no dispersion
        err += loss(y,t)
    loss_test.append(err / test.shape[0]) # average error for that epoch
#%%

fig, ax = plt.subplots(figsize=(14,10))

fig.suptitle('Cross Entropy Loss (60 dispersion)')
ax.plot(loss_train)
ax.plot(loss_test)
ax.legend(["Train", "Test"])
ax.axvline(np.argmin(loss_test), linestyle = '--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\loss_CE60.png')
fig.show()
fig.savefig(image_path)

#%%
correct_CE = 0
within1_CE = 0
within2_CE = 0

for l in range(problems.shape[0]):
    # forward pass
    input_vec = prob2vec(problems.iloc[l].Holds)
    hidden = sigmoid(np.dot(W,input_vec))
    y = softmax(np.dot(V, hidden))

    predicted_grade = np.argmax(y)
    true_grade = int(keys.loc[problems.iloc[l].Grade])
    
    diff = abs(predicted_grade - true_grade)
    if diff == 0:
        correct_CE += 1
    elif diff == 1:
        within1_CE += 1
    elif diff == 2:
        within2_CE += 1

within1_CE += correct_CE
within2_CE += within1_CE
correct_CE /= problems.shape[0]
within1_CE /= problems.shape[0]
within2_CE /= problems.shape[0]

fig, ax = plt.subplots(figsize=(12,9))
accuracies_60 = [correct_CE, within1_CE, within2_CE]
fig.suptitle('Cross Entropy Total Accuracy, dispersion .60')
ax.bar(['Correct', 'Within 1', 'Within 2'], accuracies_60, width = .5)
ax.set_ylabel('Accuracy (%)')
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\accuracy_60.png')
fig.show()
fig.savefig(image_path)
# %%
pd.DataFrame(W).to_csv("tuned_matrices/ce2017W_60.csv", header = None, index = None)
pd.DataFrame(V).to_csv("tuned_matrices/ce2017V_60.csv", header = None, index = None)

# %% 90 dispersion

dispersion_rate = .90 # rate of dispersion
W = np.random.normal(0,var,size = (num_hidden,198)) # input to hidden
V = np.random.normal(0,var,size = (len(labels),num_hidden)) # hidden to output

loss_train = []
loss_test = []
for epoch in range(num_epochs):
    if epoch % 5 == 0:
        print("epoch:", epoch)
    shuffle(train)
    err = 0
    for l in range(train.shape[0]): # number of problems in training set

        # forward pass
        input_vec = prob2vec(train.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = softmax(np.dot(V, hidden))
        t = grade2vec(keys.loc[train.iloc[l].Grade][1])
        disperse1(t, dispersion_rate)
        err += loss(y,t)

        # backward pass
        for j in range(len(labels)): # update V
            V[j,:] -= eta * (y[j] - t[j]) * hidden 
        
        for i in range(num_hidden):
            W[i,:] -= eta * np.dot(y-t, V[:,i]) * hidden[i] * (1-hidden[i]) * input_vec # 198 dim vector
    loss_train.append(err / train.shape[0]) # average error for that epoch

    err = 0
    for l in range(test.shape[0]):
        # forward pass
        input_vec = prob2vec(test.iloc[l].Holds)
        hidden = sigmoid(np.dot(W,input_vec))
        y = softmax(np.dot(V, hidden))
        t = grade2vec(keys.loc[test.iloc[l].Grade][1])
        #disperse1(t, dispersion_rate) # When testing, no dispersion
        err += loss(y,t)
    loss_test.append(err / test.shape[0]) # average error for that epoch
#%%

fig, ax = plt.subplots(figsize=(14,10))

fig.suptitle('Cross Entropy Loss (90 dispersion)')
ax.plot(loss_train)
ax.plot(loss_test)
ax.legend(["Train", "Test"])
ax.axvline(np.argmin(loss_test), linestyle = '--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\loss_CE90.png')
fig.show()
fig.savefig(image_path)

#%%
correct_CE = 0
within1_CE = 0
within2_CE = 0

for l in range(problems.shape[0]):
    # forward pass
    input_vec = prob2vec(problems.iloc[l].Holds)
    hidden = sigmoid(np.dot(W,input_vec))
    y = softmax(np.dot(V, hidden))

    predicted_grade = np.argmax(y)
    true_grade = int(keys.loc[problems.iloc[l].Grade])
    
    diff = abs(predicted_grade - true_grade)
    if diff == 0:
        correct_CE += 1
    elif diff == 1:
        within1_CE += 1
    elif diff == 2:
        within2_CE += 1

within1_CE += correct_CE
within2_CE += within1_CE
correct_CE /= problems.shape[0]
within1_CE /= problems.shape[0]
within2_CE /= problems.shape[0]

fig, ax = plt.subplots(figsize=(12,9))
accuracies_90 = [correct_CE, within1_CE, within2_CE]
fig.suptitle('Cross Entropy Total Accuracy, dispersion .90')
ax.bar(['Correct', 'Within 1', 'Within 2'], accuracies_90, width = .5)
ax.set_ylabel('Accuracy (%)')
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\accuracy_90.png')
fig.show()
fig.savefig(image_path)
# %%
pd.DataFrame(W).to_csv("tuned_matrices/ce2017W_90.csv", header = None, index = None)
pd.DataFrame(V).to_csv("tuned_matrices/ce2017V_90.csv", header = None, index = None)


# %%
