from scipy.io import arff
import FLAME as flm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# ************************** Functions **************************

# Function for creating a scatter plot with class labels
def scatter_plot(x, y, labels, cluster_labels, title):
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'cyan', 'magenta', 'black'][:len(cluster_labels)]
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=labels, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0, max(labels), max(labels) / float(len(cluster_labels)))
    cb.set_ticks(loc)
    cb.set_ticklabels(cluster_labels)
    cb.ax.set_ylabel('Clusters', rotation=270)

    plt.xlabel('Attr 1')
    plt.ylabel('Attr 2')
    plt.title(title)
    plt.show()


# Trye many k for k-nearest neighbor in FLAME to get best result
def find_optimal_k(X, y_true):
    n = 214
    rng = range(51, n)
    accuracies = []
    n_clusters = []
    max = (0,0)
    for i in rng:
        print(i)
        flame = flm.FLAME(k_neighbors=i)
        flame.cluster(X)
        y_pred = flame.single_memberships
        acc = np.sum(y_true == y_pred) / len(X)
        accuracies.append(acc)
        n_clusters.append(flame.num_clusters-1)
        if acc > max[1]:
            max = (i, acc)

    fig, ax1 = plt.subplots()
    plt.title("FLAME accuracy and # of clusters for dataset 2 with $k$ = 51 to {}".format(n))
    color = 'tab:red'
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(rng, accuracies, color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('# of clusters', color=color)
    ax2.plot(rng, n_clusters, color=color, label='# clusters')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.1, 0.4, 0.5, 0.5))
    plt.show()

    print("Maximum accuracy is {} at k={}".format(max[1], max[0]))


# Execute FLAME clustering and return FLAME object
def execute_flame(k, X):
    f = flm.FLAME(k_neighbors=k, logging=True)
    f.cluster(X)
    return f


# Get distinct list of class labels for plotting
def get_cluster_labels(f):
    clstr_labels = sorted(list(set(f.single_memberships)))
    if len(f.outliers) > 0:
        clstr_labels[-1] = 'Outliers'
    return clstr_labels

# ************************** 1. Load the datasets as NumPy array **************************


# Dataset 1
D = arff.loadarff('data.arff')
data = D[0]
n_samples = len(data)
R = data.astype([('x', '<f8'), ('y', '<f8'), ('CLASS', '<f8')]).view('<f8')
F = R.reshape(n_samples, 3)
X = F[:,:2]
y_true = F[:,2].astype('int32')  # True class labels

# Dataset 2
D2 = arff.loadarff('data2.arff')
data2 = D2[0]
n_samples2 = len(data2)
R2 = data2.astype([('a0', '<f8'), ('a1', '<f8'), ('class', '<f8')]).view('<f8')
F2 = R2.reshape(n_samples2, 3)
X2 = F2[:,:2]
y_true2 = F2[:,2].astype('int32')  # True class labels

# Map original true class labels of dataset 1 to labels in ascending order to be able to compare to predicted labels
true_class_0 = np.where(y_true == 0)
true_class_1 = np.where(y_true == 1)
true_class_2 = np.where(y_true == 2)
true_class_3 = np.where(y_true == 3)
true_class_4 = np.where(y_true == 4)
y_true[true_class_2] = 0
y_true[true_class_4] = 1
y_true[true_class_3] = 2
y_true[true_class_1] = 3
y_true[true_class_0] = 4

# find_optimal_k(X, y_true)
# find_optimal_k(X2, y_true2)

# ************************** 2. Call the FLAME clustering algorithm **************************

# Dataset 1
flame_data1 = execute_flame(58, X)
data1_y_pred = flame_data1.single_memberships
clstr_labels1 = get_cluster_labels(flame_data1)

# Dataset 2
flame_data2 = execute_flame(52, X2)
data2_y_pred = flame_data2.single_memberships
clstr_labels2 = get_cluster_labels(flame_data2)

# ************************** 3. Evaluate results **************************


def evaluate_results(y_true, f):
    y_pred = f.single_memberships
    n_samples = len(y_true)
    same = np.sum(y_true == y_pred)
    accuracy = same/n_samples
    miss = n_samples-same

    print("Misclassifications: ")
    for i, t in enumerate(y_true):
        p = y_pred[i]
        if t != p:
            print('- Sample {} should be in cluster {} but was predicted to be in {}'.format(i, t, p))
            print('     Fuzzy membership for predicted cluster {}: {}'.format(p, f.fuzzy_memberships[i, p]))
            print('     Fuzzy membership for true cluster {}: {}'.format(t, f.fuzzy_memberships[i, t]))

    print('Results:\n- Total # of samples: {}\n- Accuracy: {}%\n- # of misclassifications: {}'.format(n_samples, accuracy*100, miss))


# Dataset 1
evaluate_results(y_true, flame_data1)
scatter_plot(X[:, 0], X[:, 1], y_true, list(set(y_true)), 'Scatter plot with true labels for dataset 1')
scatter_plot(X[:, 0], X[:, 1], data1_y_pred, clstr_labels1, 'Scatter plot with predicted FLAME labels for dataset 1')

# Dataset 2
evaluate_results(y_true2, flame_data2)
scatter_plot(X2[:, 0], X2[:, 1], y_true2, list(set(y_true2)), 'Scatter plot with true labels for dataset 2')
scatter_plot(X2[:, 0], X2[:, 1], data2_y_pred, clstr_labels2, 'Scatter plot with predicted FLAME labels for dataset 2')


# ************************** 4. Execute standard k-means on same data **************************
def evaluate_kmeans_results(y_true, y_pred):
    n_samples = len(y_true)
    same = np.sum(y_true == y_pred)
    accuracy = same/n_samples
    miss = n_samples-same

    print("Misclassifications: ")
    for i, t in enumerate(y_true):
        p = y_pred[i]
        if t != p:
            print('- Sample {} should be in cluster {} but was predicted to be in {}'.format(i, t, p))
    print('Results:\n- Total # of samples: {}\n- Accuracy: {}%\n- # of misclassifications: {}'.format(n_samples, accuracy*100, miss))


# Dataset 1
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
y_pred_kmeans1 = kmeans.labels_

# Map original k-means class labels of dataset 1 to labels in ascending order to be able to compare
kmeans_class_0 = np.where(y_pred_kmeans1 == 0)
kmeans_class_1 = np.where(y_pred_kmeans1 == 1)
kmeans_class_2 = np.where(y_pred_kmeans1 == 2)
kmeans_class_3 = np.where(y_pred_kmeans1 == 3)
kmeans_class_4 = np.where(y_pred_kmeans1 == 4)
y_pred_kmeans1[kmeans_class_0] = 3
y_pred_kmeans1[kmeans_class_1] = 4
y_pred_kmeans1[kmeans_class_2] = 2
y_pred_kmeans1[kmeans_class_3] = 0
y_pred_kmeans1[kmeans_class_4] = 1

evaluate_kmeans_results(y_true, y_pred_kmeans1)
scatter_plot(X[:, 0], X[:, 1], y_pred_kmeans1, list(set(y_pred_kmeans1)), 'Scatter plot with predicted k-means labels for dataset 1')
scatter_plot(X[:, 0], X[:, 1], y_true, list(set(y_true)), 'Scatter plot with true labels for dataset 1')

# Dataset 2
kmeans2 = KMeans(n_clusters=4, random_state=0).fit(X2)
y_pred_kmeans2 = kmeans2.labels_

# Map original k-means class labels of dataset 2 to labels in ascending order to be able to compare
kmeans2_class_0 = np.where(y_pred_kmeans2 == 0)
kmeans2_class_1 = np.where(y_pred_kmeans2 == 1)
kmeans2_class_2 = np.where(y_pred_kmeans2 == 2)
kmeans2_class_3 = np.where(y_pred_kmeans2 == 3)
y_pred_kmeans2[kmeans2_class_0] = 2
y_pred_kmeans2[kmeans2_class_1] = 1
y_pred_kmeans2[kmeans2_class_2] = 0
y_pred_kmeans2[kmeans2_class_3] = 3

evaluate_kmeans_results(y_true2, y_pred_kmeans2)
scatter_plot(X2[:, 0], X2[:, 1], y_pred_kmeans2, list(set(y_pred_kmeans2)), 'Scatter plot with predicted k-means labels for dataset 2')
scatter_plot(X2[:, 0], X2[:, 1], y_true2, list(set(y_true2)), 'Scatter plot with true labels for dataset 2')