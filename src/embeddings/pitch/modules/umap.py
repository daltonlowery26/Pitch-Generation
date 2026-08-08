import numpy as np
import torch
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from torch import nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def relative_distance(batch_features, n_neighbors):
    kmu = torch.tensor(batch_features["kmu"], device=device)
    n_samples = kmu.shape[0]
    chunk_size = 800
    knn_dists_list = []
    knn_indices_list = []

    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)

        # slice scaled data
        batch_data = kmu[i:end_i]

        # compute eucd dist
        dists_chunk = torch.cdist(batch_data, kmu, p=2)

        # find neighbors
        k_dists, k_indices = torch.topk(
            dists_chunk, k=n_neighbors + 1, dim=1, largest=False
        )

        # square dist
        k_dists = k_dists**2

        knn_dists_list.append(k_dists.cpu())
        knn_indices_list.append(k_indices.cpu())

    dists = torch.cat(knn_dists_list)
    indices = torch.cat(knn_indices_list)

    return dists, indices


def sigmaAndRho(distances, k, n_iter=32, bandwith=1.0):
    # target entropy for the distribution based on k neighbors
    target = np.log2(k) * bandwith

    # true nearest neighbor for every single point
    rhos = distances[:, 1]

    # binary search
    lo = np.zeros(distances.shape[0])
    hi = np.full(distances.shape[0], np.inf)
    mid = np.ones(distances.shape[0])

    for _ in range(n_iter):
        # (distance between i/j minus nn ) / varience of i
        val = np.maximum(distances[:, 1:] - rhos[:, None], 0.0) / mid[:, None]

        # sum of weights
        p_sum = np.sum(np.exp(-val), axis=1)

        # binary search
        diff = p_sum - target
        mask_low = diff < 0
        mask_high = diff > 0

        # increase sigma
        lo[mask_low] = mid[mask_low]
        inf_mask = np.isinf(hi)
        mid[mask_low & inf_mask] *= 2
        mid[mask_low & ~inf_mask] = (
            lo[mask_low & ~inf_mask] + hi[mask_low & ~inf_mask]
        ) / 2

        # decrease sigma
        hi[mask_high] = mid[mask_high]
        mid[mask_high] = (lo[mask_high] + hi[mask_high]) / 2
    return rhos, mid


def adjMatrix(indices, distances, sigmas, rhos):
    n_samples = indices.shape[0]
    n_neighbors = indices.shape[1]  # k nearest neighbors

    # coo matrix, stores row indices, column indices, and values, rest is zeros\
    rows = np.repeat(np.arange(n_samples), n_neighbors - 1)
    cols = indices[:, 1:].flatten()
    dists = distances[:, 1:].flatten()

    # align paramters
    rhos_flat = np.repeat(rhos, n_neighbors - 1)
    sigmas_flat = np.repeat(sigmas, n_neighbors - 1)

    # weight of edge, how close a connection is
    val = np.maximum(dists - rhos_flat, 0.0) / sigmas_flat
    weights = np.exp(-val)

    # the probablity j is similar from i, how close i to j from i (csr more efficent in transpose and additon)
    p_cond = coo_matrix((weights, (rows, cols)), shape=(n_samples, n_samples)).tocsr()

    # tranpose matrix to view from other points persepective
    tP_cond = p_cond.transpose()

    # strength of relationship if one side thinks string assume it is
    prod = p_cond.multiply(tP_cond)
    p_sym = p_cond + tP_cond - prod

    return p_sym


def ab(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


# create graph
def umapGraph(df, n_neighbors):
    # nearest points
    dists, indices = relative_distance(df, n_neighbors)

    # to cpu
    dists = dists.cpu().numpy()
    indices = indices.cpu().numpy()

    # sigmas and rhos, sparse graph
    rhos, sigmas = sigmaAndRho(dists, n_neighbors)
    graph = adjMatrix(
        indices, dists, sigmas, rhos
    )  # how likely a point is to be connected

    del dists, indices

    # from sparse matrix
    graph = graph.tocoo()
    graph.eliminate_zeros()

    return graph


# umap sampling
class UmapDataset(Dataset):
    def __init__(self, graph):
        # save coo
        self.graph_rows = graph.row
        self.graph_cols = graph.col
        self.weights = graph.data

        # first batch of indices
        self.heads = None
        self.tails = None
        self.resample()

    def resample(self):
        # probablity of including
        counts = np.floor(self.weights).astype(np.int32)
        fractional = self.weights - counts
        counts += np.random.binomial(1, fractional)

        # indicies for epoch based on prob
        self.heads = np.repeat(self.graph_rows, counts)
        self.tails = np.repeat(self.graph_cols, counts)

        # shuffle
        perm = np.random.permutation(len(self.heads))
        self.heads = self.heads[perm]
        self.tails = self.tails[perm]

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        return self.heads[idx], self.tails[idx]


class UmapLoss(nn.Module):
    def __init__(self, minDist=0.4, spread=1.0, negSample=10):
        super().__init__()
        self.neg_sample_rate = negSample
        self.a, self.b = ab(spread, minDist)

    def forward(self, embedding_to, embedding_from):
        B = embedding_to.shape[0]
        eps = 1e-6

        # positive edges, pull connected points together
        d2_pos = ((embedding_to - embedding_from) ** 2).sum(dim=1)
        loss_pos = torch.log1p(self.a * (d2_pos + eps) ** self.b).mean()

        # negative edges, push random b*k pairs apart
        idx = torch.randint(0, B, (B, self.neg_sample_rate), device=embedding_to.device)
        neg = embedding_from[idx]
        d2_neg = ((embedding_to.unsqueeze(1) - neg) ** 2).sum(dim=-1)
        inv = 1.0 / (self.a * (d2_neg + eps) ** self.b + eps)
        loss_neg = torch.log1p(inv).mean()

        return loss_pos + loss_neg
