import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from torch.nn.modules.module import Module
from util import loadGraph, maxk, normalizeG, target_distribution, eva, load_data, knnL2
import time
import warnings

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class GNNLayer(Module):

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Linear(in_features, out_features)

    def forward(self, features, adj, active=True):
        output = self.weight(torch.mm(adj, features))
        if active:
            output = F.relu(output)
        return output


class AE(nn.Module):

    def __init__(self, enc, dec):
        super(AE, self).__init__()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(enc) - 1):
            self.enc.append(Linear(enc[i], enc[i + 1]))
        for i in range(len(dec) - 1):
            self.dec.append(Linear(dec[i], dec[i + 1]))

    def forward(self, x):

        for i in self.enc[:-1]:
            x = i(x)
            x = F.relu(x)
        z = self.enc[-1](x)
        x = z
        for i in self.dec[:-1]:
            x = F.relu(i(x))
        x_bar = self.dec[-1](x)

        return x_bar, z


class SGE(nn.Module):

    def __init__(self, enc, dec):
        super(SGE, self).__init__()
        self.enc = nn.ModuleList()
        for i in range(len(enc) - 1):
            self.enc.append(GNNLayer(enc[i], enc[i + 1]))

        self.dec = nn.ModuleList()
        for i in range(len(dec) - 1):
            self.dec.append(Linear(dec[i], dec[i + 1]))

    def forward(self, x, adj):
        for i in self.enc[:-1]:
            x = i(x, adj)
        z = self.enc[-1](x, adj, False)
        x = z

        for i in self.dec[:-1]:
            x = F.relu(i(x))
        x_bar = self.dec[-1](x)

        return x_bar, z


class Predictor(nn.Module):

    def __init__(self, enc, n_clusters):
        super(Predictor, self).__init__()
        self.l1 = Linear(enc, 2000)
        self.l3 = Linear(2000, n_clusters)

    def forward(self, x, deta=1):
        if deta == 1:
            x = self.l1(x.detach())
        else:
            x = self.l1(x)
        x2 = F.relu(x)
        x2 = self.l3(x2)

        return x2


class AGFC_Net(nn.Module):

    def __init__(self, enc, dec, n_clusters):
        super(AGFC_Net, self).__init__()

        self.ae = SGE(enc, dec)

        self.cl = Predictor(enc[-1], n_clusters)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, enc[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj=0, deta=1):

        x_bar0, z0 = self.ae(x[0], torch.eye(x[0].shape[0]).to(device))
        if adj == 0:  #Eq.1
            return x_bar0, z0

        if len(adj) == 1:  #Eq.3
            x_bar1, z1 = self.ae(x[1], adj[0])
            q = 1.0 / (1.0 + torch.sum(
                torch.pow(z1.unsqueeze(1) - self.cluster_layer, 2), 2))
            q = (q.t() / torch.sum(q, 1)).t()  #Eq.4
            cl = self.cl(z1, deta)
            cl = F.softmax(cl, 1)
            return x_bar0, z0, x_bar1, z1, q, cl

        x_bar = []
        z = []
        cl = []
        for i in range(len(x) - 1):
            x_bar1, z1 = self.ae(x[i + 1], adj[i])
            x_bar.append(x_bar1)
            z.append(z1)
            cl.append(self.cl(z1, deta))

        return cl, z0


def train(dataset):
    X = dataset.x
    Y = dataset.y
    Y = Y - Y.min()

    n0 = dataset.__len__()

    n = X.shape[0]
    n_input = X.shape[1]
    n_z = 10
    n_clusters = np.unique(Y).shape[0]

    resp = []

    data = torch.Tensor(X).to(device)

    globalstep = 0

    if name in ['dblp', 'acm', 'cite', 'amap']:
        path = 'graph/{}_graph.txt'.format(name)
        adj0 = loadGraph(path, n0, device)
    else:
        path = 'graph/{}{}_graph.txt'.format(name, 5)
        adj0 = loadGraph(path, n, device)

    model = AGFC_Net([n_input, 500, 500, 2000, n_z],
                     [n_z, 2000, 500, 500, n_input], n_clusters).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    print("parameters:", sum(i.numel() for i in model.parameters()))

    olddict = torch.load(pretrain, map_location='cpu')
    newdict = model.ae.state_dict()
    keys = []
    for key, v in newdict.items():
        keys.append(key)
    for i, (key, v) in enumerate(olddict.items()):
        newdict[keys[i]] = v
    model.ae.load_state_dict(newdict)

    model.cluster_layer.data = torch.load('data/' + name + '_center.pkl',
                                          map_location='cpu').to(device)

    with torch.no_grad():
        _, z1 = model.ae(data, torch.eye(adj0.shape[0]).to(device))

    adj = knnL2(z1.detach(), k=5, device=device)

    drop = nn.Dropout(p=maskA)

    adjs = (normalizeG(adj0, device) + normalizeG(adj, device)) / 2

    adjn = 0

    # Stage 1
    for epoch2 in range(tim):
        start = 0
        step = 1 / 2

        mask = torch.rand(data.shape[1])
        mask1 = mask < start + maskWidth
        mask1 = mask1.reshape((1, -1)).repeat(data.shape[0], 1).to(device)
        data1 = data * mask1

        if name in ['amap', 'usps']:
            x_bar0, z0, x_bar1, z1, q, cl = model([data, data], [adjs], 1)
        else:
            x_bar0, z0, x_bar1, z1, q, cl = model([data, data1], [drop(adjs)],
                                                  1)

        p = target_distribution(q.detach())  #Eq.5

        loss1 = F.kl_div(q.log(), p, reduction='batchmean')  #Eq.6
        loss2 = F.kl_div(cl.log(), p, reduction='batchmean')  #Eq.7
        loss3 = F.mse_loss(x_bar0, data)  #Eq.2

        loss = lam1 * loss1 + lam2 * loss2 + lam3 * loss3  #Eq.8

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch2 % 10 == 0:
            # update_interval
            res2 = cl.detach().cpu().numpy().argmax(1)
            resp.append(eva(Y, res2, str(epoch2), show))
            globalstep += 1

        #Graph Feedback Module
        if epoch2 > tim3 - 1:
            if (epoch2 - tim3) % T == 0:
                adjtmp = knnL2(z0.detach(), k, device=device)
                adj2 = adjtmp
            else:
                adjtmp = knnL2(z0.detach(), k, device=device)
                adj2 = adj2 + adjtmp  #Eq.10
            if (epoch2 - tim3) % T == T - 1:
                adj2 = maxk(adj2, k, device=device)  #Eq.11
                adjn = 1
                adj = (adjn * adj + adj2) / (adjn + 1)
                adjs = (normalizeG(adj0, device) + normalizeG(adj, device)) / 2
    # Stage 2
    for epoch2 in range(tim2):
        start = 0
        step = 1 / 2

        mask = torch.rand(data.shape[1])
        mask1 = mask < start + maskWidth
        mask1 = mask1.reshape((1, -1)).repeat(data.shape[0], 1).to(device)
        data1 = data * mask1
        mask2 = (mask > start + step) * (mask < start + step + maskWidth) + (
            mask < start + step + maskWidth - 1)
        mask2 = mask2.reshape((1, -1)).repeat(data.shape[0], 1).to(device)
        data2 = data * mask2

        cl, z0 = model([data, data1, data2], [drop(adjs), drop(adjs)], 0)

        PyM = torch.eye(n).to(device)

        NyM = (1 - (adjs > 0).int())

        out1 = (F.normalize(cl[0], 1)).squeeze()
        out2 = (F.normalize(cl[1], 1)).squeeze()

        dist1 = torch.exp(out1 @ out1.T / temperature)
        dist2 = torch.exp(out1 @ out2.T / temperature)
        dist3 = torch.exp(out2 @ out2.T / temperature)

        Ng = torch.cat(((NyM * dist1).sum(dim=-1) + (NyM * dist2).sum(dim=-1),
                        (NyM * dist2).sum(dim=-1) + (NyM * dist3).sum(dim=-1)),
                       dim=0)
        Pos = torch.cat(((dist2 * PyM).sum(dim=-1), (dist2 * PyM).sum(dim=-1)),
                        dim=0)
        loss5 = lam5 * (-torch.log(Pos / (Pos + Ng))).mean()  #Eq.9

        loss = loss5
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch2 % 10 == 0:
            # update_interval
            x_bar0, z0, x_bar1, z1, q, cl = model([data, data], [adjs], 1)

            res2 = cl.detach().cpu().numpy().argmax(1)
            resp.append(eva(Y, res2, str(epoch2), show))
            globalstep += 1
        #Graph Feedback Module
        if epoch2 > tim4 - 1:
            if (epoch2 - tim4) % T == 0:
                adjtmp = knnL2(z0.detach(), k, device=device)
                adj2 = adjtmp
            else:
                adjtmp = knnL2(z0.detach(), k, device=device)
                adj2 = adj2 + adjtmp  #Eq.10
            if (epoch2 - tim4) % T == T - 1:
                adj2 = maxk(adj2, k, device=device)  #Eq.11
                adjn = 1
                adj = (adjn * adj + adj2) / (adjn + 1)
                adjs = (normalizeG(adj0, device) + normalizeG(adj, device)) / 2

    resp = np.array(resp)
    return resp.max(0)


if __name__ == "__main__":

    start = time.time()
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--name', type=str, default="dblp")
    parser.add_argument('--img_name', type=str, default="dblp.png")
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--cuda_core', type=str, default='0')
    parser.add_argument('--mask_x', type=float, default=0.2)
    parser.add_argument('--mask_A', type=float, default=0.1)
    parser.add_argument('--time1', type=int, default=40)
    parser.add_argument('--time2', type=int, default=80)
    parser.add_argument('--time3', type=int, default=0)
    parser.add_argument('--time4', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--lam1', type=float, default=0.1)
    parser.add_argument('--lam2', type=float, default=0.01)
    parser.add_argument('--lam3', type=float, default=0.001)
    parser.add_argument('--lam4', type=float, default=0.0)
    parser.add_argument('--lam5', type=float, default=1.0)
    parser.add_argument('--tempe', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--show', type=int, default=1)
    parser.add_argument('--contra', type=int, default=1)

    args = parser.parse_args()

    name = args.name
    epo = args.epoch
    show = args.show

    pretrain = 'data/{}.pkl'.format(name)

    device = torch.device(
        "cuda:" + args.cuda_core if torch.cuda.is_available() else "cpu")

    dataset = load_data(name)

    temperature = 0.5
    maskX = args.mask_x
    maskA = args.mask_A
    tim = args.time1
    tim2 = args.time2
    tim3 = args.time3
    tim4 = args.time4
    lam1 = args.lam1
    lam2 = args.lam2
    lam3 = args.lam3
    lam4 = args.lam4
    lam5 = args.lam5
    imgName = args.img_name
    lr = args.lr
    seed = args.seed
    k = args.k
    T = args.T
    contra = args.contra

    if args.name == 'dblp':
        tim = 40
        tim2 = 80
        tim3 = 0
        tim4 = 0
        lam1 = 0.1
        lam2 = 0.01
        lam3 = 1
        k = 5
        T = 20
        lr = 1e-3
        maskX = 0.4
        maskA = 0.2
    if args.name == 'acm':
        tim = 40
        tim2 = 80
        tim3 = 0
        tim4 = 0
        lam1 = 0.1
        lam2 = 0.01
        lam3 = 0.001
        k = 5
        T = 20
        lr = 1e-3
        maskX = 0.2
        maskA = 0.1
    if args.name == 'amap':
        tim = 100
        tim2 = 80
        tim3 = 0
        tim4 = 0
        lam1 = 0.1
        lam2 = 0.01
        lam3 = 0.001
        k = 5
        T = 20
        lr = 1e-4
        maskX = 0.2
        maskA = 0.1
    if args.name == 'cite':
        tim = 80
        tim2 = 80
        tim3 = 0
        tim4 = 0
        lam1 = 0.1
        lam2 = 0.01
        lam3 = 0.001
        k = 5
        T = 20
        lr = 1e-4
        maskX = 0.2
        maskA = 0.01
    if args.name == 'usps':
        tim = 40
        tim2 = 80
        tim3 = 0
        tim4 = 0
        lam1 = 0.1
        lam2 = 0.1
        lam3 = 0.001
        k = 3
        T = 20
        lr = 1e-3
        maskX = 0.3
        maskA = 0.01
    if args.name == 'reut':
        tim = 100
        tim2 = 80
        tim3 = 0
        tim4 = 0
        lam1 = 0.1
        lam2 = 0.1
        lam3 = .001
        k = 5
        T = 20
        lr = 1e-4
        maskX = 0.2
        maskA = 0.01

    maskWidth = 1 - maskX

    setup_seed(seed)
    resp = train(dataset)
    print("acc:", resp[0], "ari:", resp[1], "f1:", resp[2])
    end = time.time()
    print('time:', end - start)
