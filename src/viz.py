import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.metrics import f1_score,roc_auc_score,accuracy_score

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.metrics import AUC,Accuracy

from src.tensors.metrics import F1Score

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

def show_tensor_stats(y_true,y_pred):
    metric_names = ["Accuracy","AUC","F1 Score"]
    metrics = [Accuracy(),AUC(),F1Score()]
    for m in metrics:
        m.update_state(y_true,y_pred)
    
    for n,m in zip(metric_names,metrics):
        print(f"{n}: {m.result():.4%}")

def show_history(history,val=True):
    metrics = ["Accuracy","F1_score","AUC","Loss"]

    fig, axs = plt.subplots(len(metrics), 1,figsize=(10, 8))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("Model metrics", fontsize=24)

    for ax,metric in zip(axs,metrics):

        xy = history.history[metric.lower()]

        ax.plot(xy, label=metric)

        if val:
            val_metric = f"val_{metric.lower()}"

            xy = history.history[val_metric]

            ax.plot(xy, label=val_metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()

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

def plot_density(X_train: np.ndarray, y_train: np.ndarray, criterion_1: int, criterion_2: int,w,b) -> None:
    # Plot the density of the training data
    axes = sns.jointplot(
        x=X_train[:, criterion_1],
        y=X_train[:, criterion_2],
        hue=y_train[:],
        kind="kde",
        palette=["green", "red"],
        fill=True,
        alpha=0.5,
    )
    
    # Plot the decision boundary
    x_points = np.linspace(-1, 1)
    y_points = -(w[criterion_1] / w[criterion_2]) * x_points - b / w[criterion_2]
    #x_points = x_points[np.logical_and(y_points < 1, y_points > -1)]
    #y_points = y_points[np.logical_and(y_points < 1, y_points > -1)]
    axes.ax_joint.plot(x_points, y_points, c="blue")
    
    # Set labels for axes
    axes.ax_joint.set_xlabel(f"Criterion: {criterion_1}")
    axes.ax_joint.set_ylabel(f"Criterion: {criterion_2}")

    # color = {-1:"red",1:"green"}
    # for c in range(-1,1,2):
    #     x = np.sort(np.unique(X_train[y_train==c,criterion_1]))
    #     y = np.sort(np.unique(X_train[y_train==c,criterion_2]))

    #     z = np.zeros((len(x),len(y)))
    #     for i,x_i in enumerate(x):
    #         sub_x = X_train[:,criterion_1]==x_i
    #         for j,y_j in enumerate(y):
    #             sub_y = X_train[:,criterion_2]==y_j
    #             z[i,j] = np.sum(np.logical_and(sub_x,sub_y))

    #     plt.contour(x,y,z,alpha=0.5,color=color[c])

    # x_points = np.linspace(-1, 1)
    # y_points = -(weights[criterion_1] / weights[criterion_2]) * x_points - intercept / weights[criterion_2]
    # fltr = np.logical_and(y_points < 1, y_points > -1)
    # x_points = x_points[fltr]
    # y_points = y_points[fltr]
    # plt.plot(x_points, y_points, color="blue")
    
    # plt.xlabel(f"Criterion: {criterion_1}")
    # plt.ylabel(f"Criterion: {criterion_2}")

def plot_corr(X,labels,title):
    corr = pd.DataFrame(X,columns=labels).corr()
    plt.matshow(corr,cmap="seismic")
    plt.xticks(np.arange(len(labels)),labels=labels,rotation=60)
    plt.yticks(np.arange(len(labels)),labels=labels)
    cb = plt.colorbar()
    plt.clim(-1,1)
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16);