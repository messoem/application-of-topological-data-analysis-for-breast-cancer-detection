import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble

# Для данных мы используем датасет о раке молочной железы штата Висконсин.
# считываем данные из таблицы формата csv:
df = pd.read_csv("data.csv")
feature_names = [c for c in df.columns if c not in ["id", "diagnosis"]]
df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)
X = np.array(df[feature_names].fillna(0))
y = np.array(df["diagnosis"])

#создаем numpy-массивы для хранения полученных из таблицы данных


#  создаем индивидуальную одномерную линзу с помощью изоляционного леса
#Управляет псевдослучайностью выбора признаков и значений разделения для каждого шага ветвления и каждого дерева в лесу
model = ensemble.IsolationForest(random_state=1729)

model.fit(X)
lens1 = model.decision_function(X).reshape((X.shape[0], 1))

# Создаем еще одну одномерную линзу с L2-нормой

mapper = km.KeplerMapper(verbose=3)
lens2 = mapper.fit_transform(X, projection="l2norm")

# Объединим обе линзы, чтобы создать двумерную линзу: изоляционный лес и  L^2-норму.
lens = np.c_[lens1, lens2]

# Построение симплициального комплекса
graph = mapper.map(
    lens,
    X,
    cover=km.Cover(n_cubes=15, perc_overlap=0.4),
    clusterer=sklearn.cluster.KMeans(n_clusters=2, random_state=1618033)
)

# Визуализация
mapper.visualize(graph,
    path_html="breast-cancer.html",
    title="Wisconsin Breast Cancer Dataset",
    custom_tooltips=y
)


# Визуализация с несколькими цветовыми функциями
mapper.visualize(
    graph,
    path_html="breast-cancer-multiple-color-functions.html",
    title="Wisconsin Breast Cancer Dataset",
    custom_tooltips=y,
    color_values=lens,
    color_function_name=["Isolation Forest", "L2-norm"]
)


# Visualization with multiple node color functions
mapper.visualize(
    graph,
    path_html="breast-cancer-multiple-node-color-functions.html",
    title="Wisconsin Breast Cancer Dataset",
    custom_tooltips=y,
    node_color_function=["mean", "std", "median", "max"]
)

# Visualization showing both multiple color functions, and also multiple node color functions
mapper.visualize(
    graph,
    path_html="breast-cancer-multiple-color-functions-and-multiple-node-color-functions.html",
    title="Wisconsin Breast Cancer Dataset",
    custom_tooltips=y,
    color_values=lens,
    color_function_name=["Isolation Forest", "L2-norm"],
    node_color_function=["mean", "std", "median", "max"]
)


import matplotlib.pyplot as plt

km.draw_matplotlib(graph)
plt.savefig("mygraph1.pdf")
