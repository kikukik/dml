"""
@author: K. Kersting, Z. Yu, J.Czech
Machine Learning Group, TU Darmstadt
"""
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import graphviz
from sklearn.metrics import accuracy_score


def fit_dt_classifier(X_train: np.ndarray, y_train: np.ndarray) -> tree.DecisionTreeClassifier:
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(X_train,y_train)
    return clf

def get_test_accuracy(clf, X_test: np.ndarray, y_test: np.ndarray) -> float:
    missclassifications=0
    for idx,d in enumerate(X_test):
        predict=clf.predict(d.reshape(1,-1))
        if predict!=y_test[idx]:
            missclassifications=missclassifications+1    
    return 1-missclassifications/len(X_test)



def export_tree_plot(clf, filename: str):
    tree.plot_tree(clf)
    dot_data=tree.export_graphviz(clf,out_file=None)
    graph=graphviz.Source(dot_data)
    graph.render(filename)
    return

def main():
    # for reproducibility
    np.random.seed(42)

    # load data
    X_data = np.loadtxt(open('./PtU/FileName_Fz_raw.csv', 'r'), delimiter=",", skiprows=0)
    y_data = np.loadtxt(open('./PtU/FileName_Speed.csv', 'r'), delimiter=",", skiprows=0)

    # down sample the data
    X_sample = X_data[:, ::100]
    print("X_data.shape:", X_data.shape)
    print("y_data.shape:", y_data.shape)

    # split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_data, test_size=0.2)

    # train
    clf = fit_dt_classifier(X_train, y_train)
    # predict
    acc = get_test_accuracy(clf, X_test, y_test)
    print('Test Accuracy:', acc)

    print("predict_proba:", clf.predict_proba(X_test))

    # plot tree
    export_tree_plot(clf, "classification_tree")


if __name__ == '__main__':
    main()
