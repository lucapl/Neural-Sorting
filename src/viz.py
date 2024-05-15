import matplotlib.pyplot as plt

from collections import Counter

from sklearn.metrics import f1_score,roc_auc_score,accuracy_score

import numpy as np

import tensorflow as tf

def visualize_data(X,y):
    cmap = ["red","blue"]
    #colors = tuple(map(lambda c:cmap[int(c)-1],data["class"]))
    #for i,col in enumerate(data.drop(columns="class")):
    features = X.shape[1]
    fig,axs = plt.subplots(1,2,figsize=(10,6))
    fig.tight_layout()
    for i,x in enumerate(X):
        
        _y = np.copy(x)
        _x = np.arange(features,dtype=np.float64)
        c = int(y[i])
        color = cmap[c]
        #_y = np.copy(X[:,])
        _y += np.random.normal(0,0.025,len(_y))
        _y = np.clip(_y,0,1)
        #_x = np.array([i for _ in range(len(_y))]).astype(np.float64)
        _x += np.random.normal(0,0.1,len(_x))
        axs[c].plot(_x,_y,alpha=0.05,color=color)
        axs[c].set_ylabel(f"Criterion value")
        axs[c].set_xlabel(f"Criterions")
        axs[c].set_yticks(np.arange(1,step=0.0833333))
        axs[c].set_xticks(np.arange(features,step=1))

def show_class_counts(y):
    for c,count in Counter(y).items():
        print(f"Class {c} occurences: {count}")

def show_combinations(data):
    _combs = 1
    for col in data:
        _combs *= len(np.unique(data[col]))

    print("Possible combinations of data:",_combs)
    print("Number of alternatives:",len(data))

def show_stats(model,X,y):
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y,y_pred):.4%}")
    print(f"F1 score: {f1_score(y,y_pred):.4%}")
    auc = np.dot(X,model.coef_.T)
    print(f"AUC: {roc_auc_score(y,auc):.4%}")

def show_history(history):
    metrics = ["Accuracy","F1_score","AUC","Loss"]

    fig, axs = plt.subplots(len(metrics), 1,figsize=(10, 8))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("Model metrics", fontsize=24)

    for ax,metric in zip(axs,metrics):

        xy = history.history[metric.lower()]

        ax.plot(xy, label=metric)
        ax.plot(xy, label=metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)

def show_monotone_blocks(uta,features):

    fig,axs = plt.subplots(1,features,figsize=(15,6))
    fig.tight_layout()
    for i in range(features):
        f = uta.get_layer(name=f"monotone_block_{i}").call
        x = np.arange(0,1+0.05,0.05)
        y = [np.max(f(tf.reshape([x_i],(1,1)))) for x_i in x]

        axs[i].plot(x,y)
        axs[i].set_xlabel("criterion value")
        axs[i].set_ylabel("criterion utility")
        axs[i].set_yticks(x)

def show_criteria_weights(uta_model):
    for i,weight in enumerate(uta_model.uta.get_layer("criteria_weights").get_weights()[0]):
        print("Weight of criterion",i,":",weight[0])