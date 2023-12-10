from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import numpy as np
from sklearn import metrics
from munkres import Munkres
from torch.utils.data import Dataset
import scipy.sparse as sp


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    X = X.transpose(1, 0)
    Y = Y.transpose(1, 0)
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def loadGraph(path, n, device):
    edges = np.genfromtxt(path, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n),
                        dtype=np.float32)
    adj = torch.from_numpy(adj.toarray()).to(device)
    return adj


def knnL2(data, k=5, A=None, alpha=1, data2=None, device=torch.device("cpu")):
    n = data.shape[0]
    features = data
    if A is not None:
        dist = distance(features, features) - alpha * A
    elif data2 is not None:
        dist = distance(features, features) + alpha * distance(data2, data2)
    else:
        dist = distance(features, features)
    adj = torch.zeros(dist.shape).to(device)
    indexy = dist.argsort()[:, 1:k + 1]
    indexx = torch.arange(n).reshape((n, 1)).repeat(1, k)

    adj[indexx, indexy] = 1
    return adj


def maxk(mat, k=5, device=torch.device("cpu")):
    n = mat.shape[0]
    adj = torch.zeros(mat.shape).to(device)
    indexy = (-mat).argsort()[:, :k]
    indexx = torch.arange(n).reshape((n, 1)).repeat(1, k)
    adj[indexx, indexy] = 1
    return adj


def normalizeG(adj, device=torch.device("cpu")):
    adjn = adj + adj.T * (adj.T > adj) - adj * (adj.T > adj)
    adjn = adjn + torch.eye(adjn.shape[0]).to(device)
    adjn = normalize(adjn)
    return adjn


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    cost = np.zeros((numclass1, numclass1), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        if indexes[i][1] < numclass2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
            new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro, new_predict


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def eva(y_true, y_pred, epoch=0, print1=0):
    acc, f1, _ = cluster_acc(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    if print1 == 1:
        print(epoch, ':acc {:.4f}'.format(acc), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, ari, f1


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx


class load_data(Dataset):

    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset),
                            dtype=float).astype(np.float32)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))
