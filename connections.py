#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.utils import shuffle
from scipy.cluster.hierarchy import linkage, dendrogram
#%%
with open('C:\\Users\\nathan\\Desktop\\hw_21-1\\dl_cse\\2021-1\\project\\moon17.json') as f:
    data = json.load(f)

# Preprocessing
problems = pd.DataFrame(data)
problems = problems.drop(["DateInserted", 'Method', 'MoonBoardConfigurationId', 'Setter',
       'Rating', 'Holdsetup', 'IsAssessmentProblem'])
problems = problems.T


problems = problems[problems['Repeats']>0] # only ones that are repeated

#%%
problems = problems.reset_index()
all_moves = pd.Series([])
for i in range(len(problems)):
    moves = []
    for m in range(len(problems.iloc[i].Moves)):
        moves.append(problems.iloc[i].Moves[m]['Description'])
    all_moves[i] = moves
problems.insert(2, "Holds", all_moves)
#%%

grades = list(problems.Grade.unique())
grades.sort()
moon17 = {elem : pd.DataFrame for elem in grades}

for key in moon17.keys():
    moon17[key] = problems[:][problems.Grade == key].reset_index(drop=True)
    moon17[key].set_index('Name', inplace=True)
# %% Fixed parameters
num_holds = 11*18
#%% Changeable parameters
num_epochs = 200
eta = 0.02

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

def hold2int(hold): # input hold, output one-hot vector of length num_holds
    pos = (ord(hold[0])-65) * 18 # letter A->0, ..., K-> 10
    if len(hold) == 2:
        pos += int(hold[1]) # number after the letter
    else: # two digit number
        pos += int(hold[1])*10 + int(hold[2])
    return pos-1

def prob2vec(prob):
    vec = np.zeros(num_holds)
    for hold in prob:
        vec += hold2vec(hold)
    return vec

def grade2vec(grade): # keys.loc[train.iloc[i].Grade][1] returns integer
    vec = np.zeros(len(grades))
    vec[grade] = 1
    return vec

def neighbors(Adj, i):
    return np.where(Adj[i] > 0)[0] # where the Adjacency column is nonzero

def int2hold(num): # input number, get hold
    letter = chr(num//18 + 65)
    pos = str((num % 18)+1)
    return letter+pos
# %%
keys = pd.DataFrame(moon17.keys())
keys = pd.concat([keys,pd.Series(range(len(keys)))], axis = 1 )
keys.columns = [0,1]
keys = keys.set_index([0])


# %%
Adjacency = np.zeros((num_holds, num_holds))

for l in range(problems.shape[0]):

    problem = problems.iloc[l].Holds
    for i in range(len(problem)-1):
        for hold in problem[i+1:]: # tail for that problem
            Adjacency[:,hold2int(problem[i])] += hold2vec(hold)
Adjacency = Adjacency + Adjacency.transpose()
#%%
def empirical(j,i): # j given i
    return Adjacency[i,j] / np.sum(Adjacency[i])
# %% grade distribution

for grade in grades:
    print(grade, moon17[grade].shape[0])
# %% 
# Try making a 6A+ problem

# moon17['6A+'] = moon17['6A+'].reset_index()
# all_moves = pd.Series([])
# for i in range(len(moon17['6A+'])):
#     moves = []
#     for m in range(len(moon17['6A+'].iloc[i].Moves)):
#         moves.append(moon17['6A+'].iloc[i].Moves[m]['Description'])
#     all_moves[i] = moves
# moon17['6A+'].insert(2, "Holds", all_moves)

Adjacency6Aplus = np.zeros((num_holds, num_holds))

for l in range(moon17['6A+'].shape[0]):

    problem = moon17['6A+'].iloc[l].Holds
    for i in range(len(problem)-1):
        for hold in problem[i+1:]: # tail for that problem
            Adjacency6Aplus[:,hold2int(problem[i])] += hold2vec(hold)
Adjacency6Aplus = Adjacency6Aplus + Adjacency6Aplus.transpose()


Adjacency6Aplus


# # %% Initialize
# var = 1
# embed = np.random.normal(0, var, (num_holds, 2))
# context = np.random.normal(0, var, (num_holds, 2))
# loss_array = []
# #%% Before
# import os

# fig, ax = plt.subplots(figsize=(8,6))
# df = pd.DataFrame(embed, columns = ['x', 'y'])
# df['index_'] = range(198)

# xmax, ymax = embed.max(axis=0) + 0.2
# xmin, ymin = embed.min(axis=0) - 0.2

# fig.suptitle('Initial')
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
# for i in range(198):
#     ax.text(embed[i,0], embed[i,1], str(int2hold(df['index_'][i])), fontdict={'weight':'bold', 'size':12})

# image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\line2_before.png')
# fig.show()
# fig.savefig(image_path)

# #%% 

# # scale
# #Adjacency6Aplus = Adjacency6Aplus / np.max(Adjacency6Aplus)

# # Test if some holds do not get used
# # empty = []
# # for i in range(197):
# #     if np.where(Adjacency6Aplus[i]==0)[0] == np.array(range(197)):
# #         empty.append(i)
# #%% Method 1: no weights
# unweighted = pd.DataFrame(Adjacency6Aplus) 

# unweighted = np.array(unweighted.astype(bool).astype(int))
# epsilon = 1e-100 # prevent log(0)
# for epoch in range(30):
    
#     if epoch % 3 == 0:
#         print("Epoch:", epoch)
    
#     loss = 0
#     for i in range(num_holds):
#         for j in neighbors(unweighted, i):
#             denom = 0
#             numer = 0
#             for k in range(num_holds):
#                 temp = np.exp(np.dot(context[k], embed[i]))
#                 numer += np.dot(temp, context[k])
#                 denom += temp
#             weight = unweighted[i,j]
#             update_embed =  (numer/denom - context[j])
#             update_context =  np.exp(np.dot(context[j], embed[i])) / denom
#             loss -= weight * np.log(update_context + epsilon)
#             # simultaneous update
#             context[j] -= eta * weight * update_context
#             embed[j] -= eta * weight * update_embed
#         loss_array.append(loss)    
#         loss = 0
        
# # %% After

# fig, ax = plt.subplots(figsize=(8,6))
# df = pd.DataFrame(embed, columns = ['x', 'y'])
# df['index_'] = range(198)

# xmax, ymax = embed.max(axis=0) + 0.2
# xmin, ymin = embed.min(axis=0) - 0.2

# fig.suptitle('After Line 2nd order')
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
# for i in range(198):
#     ax.text(embed[i,0], embed[i,1], str(int2hold(df['index_'][i])), fontdict={'weight':'bold', 'size':12})

# image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\line2_after.png')
# fig.show()
# fig.savefig(image_path)

# # %% negative sampling
# min = 0
# max = 200
# for i in range(197):
#     if len(np.where(Adjacency6Aplus[i]==0)[0]) > min:
#         min = len(np.where(Adjacency6Aplus[i]==0)[0])
#         argmin = i
#     if len(np.where(Adjacency6Aplus[i]==0)[0]) < max:
#         max = len(np.where(Adjacency6Aplus[i]==0)[0])
#         argmax = i

# %% Hierarchical clustering (dendrogram)
labelList = []
for i in range(198):
    labelList.append(int2hold(i))
linked = linkage(Adjacency6Aplus, method = 'single') #single == min, complete == max, average, median


plt.figure(figsize = (50, 30))
d = dendrogram(linked, 
           orientation = 'right',
           count_sort = 'ascending',
           distance_sort = 'ascending',
           labels = labelList,
           leaf_font_size = 10,
           color_threshold=370) # cutoff line
plt.show()

d['ivl'][0:50]

#%% embedding holds

k = 2 #dimension
eta = 0.02 #learning rate
epochs = 200
negs = 5

loss_list = []

num_node = Adjacency6Aplus.shape[0]
nodes = np.eye(num_node)

#Initialize weight matrices
W = np.random.normal(0,1,size = (num_node,k)) # W matrix in Skip-Gram
Context = np.random.normal(0,1,size = (num_node,k)) # W' matrix in Skip-Gram
max_connections = Adjacency6Aplus.shape[1]

#%% functions
def softmax(vec):
    exp = np.exp(vec) 
    return exp / np.sum(exp)

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def p1(vec1, vec2):
    return sigmoid(np.dot(vec1, vec2))

def dEdc(a,b):
    denom = 0
    numer = W[b,:] * np.exp(np.dot(W[b,:], Context[a,:]))
    for j in range(num_node):
        denom += np.exp( np.dot(W[b,:], Context[j,:]))
    return -W[b,:] + numer/denom

def neighbors(i):
    return np.where(Adjacency[i] > 0)[0]

def NegativeSamples(i):
    nhbrs = np.append(i, neighbors(i)) # must append itself
    
    probs = np.delete(probs1, nhbrs)
    probs = probs / np.sum(probs) # standardize to match definition of probability

    return np.random.choice(np.delete(np.arange(0, num_node), nhbrs), negs, p=probs, replace=False)


#%% Negative sampling probs
probs1 = [np.sum(Adjacency[k])**(3/4) for k in range(num_node)]
probs1 = probs1 / np.sum(probs1)
#%% Training


for e in range(epochs):
    loss = 0
    for i in range(num_node):
        
        
        for j in neighbors(i):
            p = p1(Context[j,:], W[i,:])
            C = Context[j,:] # simultaneous update
            Context[j,:] -= eta * (p-1) * W[i,:]
            W[i,:] -= eta * (p-1) * C # simultaneous update
            loss -= np.log(p)

        
        # for k in NegativeSamples(i):
        #     p = sigmoid(np.dot(Context[k,:], W[i,:]))
        #     Context[k,:] -= eta * p * W[i,:]
        #     W[i,:] -= eta * p * Context[k,:]

        #     loss -= np.log(p1(-Context[k,:], W[i,:]))
    
    loss_list.append(loss)
    
    loss = 0  


#%%

# for e in range(epochs):
    
#     loss = 0
#     for i in range(num_node):
        
        
#         for j in neighbors(i):
#             p = p1(Context[j,:], W[i,:])
#             C = Context[j,:] # simultaneous update
#             Context[j,:] -= eta * (p-1) * W[i,:]
#             W[i,:] -= eta * (p-1) * C # simultaneous update
#             loss -= np.log(p)

        
#         for k in NegativeSamples(i):
#             p = sigmoid(np.dot(Context[k,:], W[i,:]))
#             Context[k,:] -= eta * p * W[i,:]
#             W[i,:] -= eta * p * Context[k,:]

#             loss -= np.log(p1(-Context[k,:], W[i,:]))
    
#     loss_list.append(loss)
    
#     loss = 0  


# %%

# Group 1
x1 = []
y1 = []
# Group 2
x2 = []
y2 = []
# group 3
x3 =[]
y3 =[]
#group4
x4 =[]
y4 =[]
# group 5
x5 =[]
y5 =[]

reds = [0,3,5,11,13,16,20,28,35,47,49,51,55,57,61,63,67,72,74,78,81,84,87,88,89,95,97,98,104,106,109,118,130,133,137,144,146,150,153,156,158,163,170,172,175,177,178,179,181,183,185,193]
blacks = [1,4,7,26,27,30,36,42,52,56,58,59,65,69,71,75,85,90,92,96,101,107,111,115,117,120,123,128,136,139,140,142,145,147,167,174,187,188,190,196]
white = [2,10,14,15,17,18,19,21,25,31,45,48,62,70,77,79,82,91,93,94,110,113,114,121,124,126,134,135,143,148,151,155,159,165,168,180,182,192,194,196]
wood = [6,9,22,29,32,41,44,46,53,60,66,69,76,80,83,86,112,116,119,122,132,138,141,149,152,154,161,166,173,176,186,189]
# yellow = [8,12, ] #ELSE
#%%
for i in range(num_node):
    if i in reds:
        x1.append(W[i,0])
        y1.append(W[i,1])
    elif i in blacks:
        x2.append(W[i,0])
        y2.append(W[i,1])
    elif i in white:
        x3.append(W[i,0])
        y3.append(W[i,1])
    elif i in wood:
        x4.append(W[i,0])
        y4.append(W[i,1])
    else:
        x5.append(W[i,0])
        y5.append(W[i,1])
# plot Group 1

#%%

fig, ax = plt.subplots(figsize=(10,10))
# group 1
ax.scatter(x1, y1, color = "r", label = "reds")


# plot Group 2
ax.scatter(x2, y2, color = "k", label = "blacks")

# plot Group 3
ax.scatter(x3, y3, color = "gray", label = "white")

# plot Group 2
ax.scatter(x4, y4, color = "burlywood", label = "wood")

# plot Group 2
ax.scatter(x5, y5, color = "y", label = "original")

fig.suptitle('Line 1st order, with Negative Sampling')
ax.legend()
#image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\line_2.png')
fig.show()
#fig.savefig(image_path)


# %%


fig, ax = plt.subplots(figsize=(14,10))

fig.suptitle('Loss plot for Line (1st order, with Negative Sampling)')
ax.plot(loss_list)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
#image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures\\loss_2.png')
fig.show()
#fig.savefig(image_path)


#%%



#Initialize weight matrices
W = np.random.normal(0,1,size = (num_node,k)) # W matrix in Skip-Gram
Context = np.random.normal(0,1,size = (num_node,k)) # W' matrix in Skip-Gram
probs1 = [np.sum(Adjacency[k])**(3/4) for k in range(num_node)]
probs1 = probs1 / np.sum(probs1)

lossArr = []
for e in range(epochs):
    if (e % 50 == 0):
        print("Epoch: ", e)
    for i in range(num_node):
        nhbrs = neighbors(i)
        negSamples = NegativeSamples(nhbrs)
        for j in nhbrs:
            p = sigmoid(np.dot(Context[j].T, W[i]))
            Context[j] -= eta * (p - 1) * W[i]
            W[i] -= eta * (p - 1) * Context[j]
            loss += -np.log(p)
        for k in negSamples:
            p = sigmoid(np.dot(Context[k].T, W[i]))
            pl = sigmoid(np.dot(-Context[k].T, W[i]))
            Context[k] -= eta * p * W[i]
            W[i] -= eta * p * Context[k]
            loss += -np.log(pl)
    # add loss to loss array
    lossArr.append(loss)
    # renew loss
    loss = 0  
            

# %%
