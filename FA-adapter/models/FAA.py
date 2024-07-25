import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import log

from .Att import sa_layer, SELayer, CBAM
from .lshot_update import bound_update
from scipy import sparse
from models.backbones import ResNet, Conv_4
import statistics
from scipy import stats


class FAA(nn.Module):

    def __init__(self, n=5, att="SA", way=None, shots=None, resnet=False, is_pretraining=False, num_cat=None,
                 ):

        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        self.resolution = 25
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.n = n
        self.att = att
        if n == 3:
            self.sa1 = self.att_type(num_channel, att, 32)
            self.sa2 = self.att_type(num_channel, att, 16)
            self.sa3 = self.att_type(num_channel, att, 8)
        elif n == 5:
            self.sa1 = self.att_type(num_channel, att, 32)
            self.sa2 = self.att_type(num_channel, att, 16)
            self.sa3 = self.att_type(num_channel, att, 8)
            self.sa4 = self.att_type(num_channel, att, 4)
            self.sa5 = self.att_type(num_channel, att, 2)
        elif n == 7:
            self.sa1 = self.att_type(num_channel, att, 32)
            self.sa2 = self.att_type(num_channel, att, 16)
            self.sa3 = self.att_type(num_channel, att, 8)
            self.sa4 = self.att_type(num_channel, att, 4)
            self.sa5 = self.att_type(num_channel, att, 2)
            self.sa6 = self.att_type(num_channel, att, 64)
            self.sa7 = self.att_type(num_channel, att, 80)
        else:
            pass
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        if n != 0:

            t = int(abs(log(num_channel, 2) + 1) / 2)
            k = t if t % 2 else t + 1
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            self.softmax = nn.Softmax(dim=1)
        else:
            pass
        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(2), requires_grad=not is_pretraining)
        # self.fc = nn.Sequential(nn.Linear(num_channel, num_channel//80), nn.ReLU(True),
        #                         nn.Linear(num_channel//80, num_channel))

        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True)

    def att_type(self, num_channel, type="SA", groups=32):
        if type == "SA":
            return sa_layer(num_channel, groups=groups)
        if type == "SE":
            return SELayer(num_channel, reduction=groups)
        if type == "CBAM":
            return CBAM(num_channel)

    def get_feature_map(self, inp):
        batch_size = inp.size(0)
        n = self.n
        e = self.feature_extractor(inp)
        if n != 0:
            e = self.fork_att(e, n=n)
        else:
            pass

        if self.resnet:
            e = e / np.sqrt(640)

        return e.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()

    def fork_att(self, inp, n=5):
        attention_vectors = self.att_vec_generator(inp, n=n)
        feas = self.vec_modified(inp, n=n)
        e = (feas * attention_vectors).sum(dim=1)

        return e

    def vec_modified(self, inp, n=5):
        x_1 = self.sa1(inp).unsqueeze_(dim=1)
        x_2 = self.sa2(inp).unsqueeze_(dim=1)
        x_3 = self.sa3(inp).unsqueeze_(dim=1)
        if n == 3:
            feas = torch.cat([x_1, x_2, x_3], dim=1)
            return feas
        elif n == 5:
            x_4 = self.sa4(inp).unsqueeze_(dim=1)
            x_5 = self.sa5(inp).unsqueeze_(dim=1)
            feas = torch.cat([x_1, x_2, x_3, x_4, x_5], dim=1)
            return feas
        elif n == 7:
            x_4 = self.sa4(inp).unsqueeze_(dim=1)
            x_5 = self.sa5(inp).unsqueeze_(dim=1)
            x_6 = self.sa6(inp).unsqueeze_(dim=1)
            x_7 = self.sa7(inp).unsqueeze_(dim=1)
            feas = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7], dim=1)
            return feas

    def att_vec_generator(self, inp, n=5):
        vector1 = self.conv(self.avg_pool(self.sa1(inp)).squeeze(-1).transpose(-1, -2))
        vector2 = self.conv(self.avg_pool(self.sa2(inp)).squeeze(-1).transpose(-1, -2))
        vector3 = self.conv(self.avg_pool(self.sa3(inp)).squeeze(-1).transpose(-1, -2))
        if n == 3:
            attention_vectors = torch.cat([vector1, vector2, vector3], dim=1)
            attention_vectors = self.softmax(attention_vectors)
            attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
            return attention_vectors
        elif n == 5:
            vector4 = self.conv(self.avg_pool(self.sa4(inp)).squeeze(-1).transpose(-1, -2))
            vector5 = self.conv(self.avg_pool(self.sa5(inp)).squeeze(-1).transpose(-1, -2))
            attention_vectors = torch.cat([vector1, vector2, vector3, vector4, vector5], dim=1)
            attention_vectors = self.softmax(attention_vectors)
            attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
            return attention_vectors
        elif n == 7:
            vector4 = self.conv(self.avg_pool(self.sa4(inp)).squeeze(-1).transpose(-1, -2))
            vector5 = self.conv(self.avg_pool(self.sa5(inp)).squeeze(-1).transpose(-1, -2))
            vector6 = self.conv(self.avg_pool(self.sa6(inp)).squeeze(-1).transpose(-1, -2))
            vector7 = self.conv(self.avg_pool(self.sa7(inp)).squeeze(-1).transpose(-1, -2))
            attention_vectors = torch.cat([vector1, vector2, vector3, vector4, vector5, vector6, vector7], dim=1)
            attention_vectors = self.softmax(attention_vectors)
            attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
            return attention_vectors

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1) / support.size(2)

        # correspond to lambda in the paper
        lam = reg * alpha.exp() + 1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper

            sts = st.matmul(support)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(
                lam)).inverse()  # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # way, d, d

        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d

        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # way*query_shot*resolution, way

        return dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, lmd=0.1, return_support=False):
        batch_size = inp.size(0)
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]
        e1 = self.get_feature_map(inp)

        # fea B,C,HW support 100 C HW query 300 C HW
        neg_l2_dist, support, query_ = self.distance(e1, alpha, beta, d, query_shot, resolution, shot, way)

        query_ = query_.cpu()
        query_ = query_.detach().numpy()
        neg_l2_dist_cpu = neg_l2_dist.cpu()
        neg_l2_dist_cpu = neg_l2_dist_cpu.detach().numpy()
        unary = neg_l2_dist_cpu.transpose(0, 1) ** 2
        W = create_affinity(query_, knn=3)
        l = bound_update(unary, W, bound_lambda=0.5)
        l = torch.from_numpy(l).cuda()
        if return_support:
            return neg_l2_dist, l, support
        else:
            return neg_l2_dist, l

    def distance(self, fea, alpha, beta, d, query_shot, resolution, shot, way):
        support = fea[:way * shot].reshape(way, shot * resolution, d)
        query = fea[way * shot:].reshape(way * query_shot * resolution, d)
        query_ = fea[way * shot:].reshape(way * query_shot, resolution, d).mean(1)

        recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha,
                                         beta=beta)  # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().view(way * query_shot, resolution, way).mean(1)

        return neg_l2_dist, support, query_

    def meta_test(self, inp, way, shot, query_shot, lmd):

        neg_l2_dist, l = self.get_neg_l2_dist(inp=inp,
                                              way=way,
                                              shot=shot,
                                              query_shot=query_shot, lmd=lmd)

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index, l

    def forward_pretrain(self, inp):

        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size * self.resolution, self.d)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = self.get_recon_dist(query=feature_map, support=self.cat_mat, alpha=alpha,
                                         beta=beta)  # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size, self.resolution, self.num_cat).mean(1)  # batch_size,num_cat

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction

    def forward(self, inp):

        neg_l2_dist, l, support = self.get_neg_l2_dist(inp=inp,
                                                       way=self.way,
                                                       shot=self.shots[0],
                                                       query_shot=self.shots[1],
                                                       return_support=True, lmd=0.1)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction, l, support


def create_affinity(X, knn):
    N, D = X.shape
    # print('Compute Affinity ')
    nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
    dist, knnind = nbrs.kneighbors(X)

    row = np.repeat(range(N), knn - 1)
    col = knnind[:, 1:].flatten()
    data = np.ones(X.shape[0] * (knn - 1))
    W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)
    return W
