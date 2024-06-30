import torch
import math
import torch.nn.functional as F

from torch import nn
from model.functions import SparseMax
from model.network.backbone.transformer import Transformer


class Aggregator(nn.Module):
    def __init__(self, method: str = 'anchor', params=None):
        super().__init__()

        if method == 'anchor':
            self.aggregator = Anchor(**params)
        elif method == 'netvlad':
            self.aggregator = NetVLAD(**params)
        elif method == 'nextvlad':
            self.aggregator = NeXtVLAD(**params)
        elif method == 'subspace':
            self.aggregator = SubspaceCluster(**params)
        elif method == 'attention':
            self.aggregator = Attention(**params)
        elif method == 'average':
            self.aggregator = Average(**params)
        elif method == 'transform':
            self.aggregator = TransformAggregator(**params)
        else:
            raise NotImplemented('The method %s has not been implemented yet!' % method)

        self.out_channels = self.aggregator.out_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        return self.aggregator(x, mask)


class BaseAggregator(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = F.normalize(x, p=2, dim=2)
        return x


class TransformAggregator(BaseAggregator):

    def __init__(self, num_heads: int = 4, in_channels: int = 512):

        super().__init__(in_channels, in_channels)
        self.attention = Transformer(in_channels, num_heads=num_heads)
        self.token = nn.Parameter(torch.zeros(1, 1, in_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        b, n, d = x.size()
        x = super().forward(x, mask)
        # calculate the valid samples
        num_valid = mask.view(-1)

        output = []
        for i in range(b):
            data = x[i, :num_valid[i]].view(1, num_valid[i], d)
            data = torch.cat([data, self.token], dim=1)
            out = self.attention(data)
            output.append(out[:, -1])

        output = torch.cat(output, dim=0)

        return output


class SubspaceCluster(BaseAggregator):

    def __init__(self, num_space: int = 4, in_channels: int = 512, reduction: int = 8):

        super().__init__(in_channels, in_channels)
        self.num_space = num_space
        self.subspace = nn.Parameter(data=torch.randn(num_space, in_channels // reduction, in_channels,
                                                      dtype=torch.float), requires_grad=True)
        self.sparse_max = SparseMax(dim=2)
        self.scale = 1
        self.count = 0
        self.local_loss = 0

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        b, n, d = x.size()
        x = super().forward(x, mask)
        # calculate the valid samples
        num_valid = mask.view(-1)
        # preprocess the project_matrix to make its digital range be reasonable
        project_matrix = torch.bmm(self.subspace.permute(0, 2, 1), self.subspace)
        self.count += b
        self.scale = 1.718 * math.exp(-self.count / 3e4) - 0.718
        project_matrix = project_matrix / (d * self.scale)
        # calculate the projection value for each sample in each subspace
        projection = torch.matmul(project_matrix.view(1, 1, self.num_space, d, d), x.view(b, n, 1, d, 1))
        projection = projection.view(b, n, self.num_space, d).norm(dim=3)
        probability = self.sparse_max(projection)
        max_value, max_indices = probability.max(dim=2)

        weights = torch.zeros((b, n), dtype=torch.float, device='cuda')
        for i in range(b):
            index = torch.bincount(max_indices[i, :num_valid[i]]).argmax()
            weights[i] = probability[i, :, index]

        if self.training:
            cluster_loss = (1 - max_value).mean()
            try:
                rank_loss = 4 * torch.linalg.norm(project_matrix, ord='nuc', dim=(1, 2)).mean() / d
            except RuntimeError:
                rank_loss = 0
            loss = cluster_loss + rank_loss
            # print('Subspace loss: %f | Cluster loss: %f | Rank loss: %f'
            #       % (loss.item(), cluster_loss.item(), rank_loss.item()))
            self.local_loss = loss

        feature = (x * weights.view(b, n, 1)).sum(1)

        return feature


class Anchor(BaseAggregator):

    def __init__(self, num_cluster: int = 2, in_channels: int = 512, regular: int = 0, max_type: str = 'soft'):
        """
        This is a anchor aimed in learning cluster automatically.

        :param num_cluster: The number of anchor need to learn
        :param in_channels: The dimensions of the feature
        :param regular: The regular type. Specifically, 0 represents use the BCE loss only, 1 represents use the BCE
               plus the Cluster loss. 2 represents use the BCE loss plus the Divergence loss. 3 represents use the BCE,
               Cluster loss and Divergence loss together.
        :rtype: object
        """

        super().__init__(in_channels, in_channels)
        self.num_anchor = num_cluster
        self.anchors = nn.Parameter(torch.rand(num_cluster, in_channels, dtype=torch.float), requires_grad=True)
        self.sparse_max = SparseMax(dim=1) if max_type == 'sparse' else nn.Softmax(dim=1)
        self.regular = regular
        self.local_loss = torch.tensor(0, dtype=torch.float, requires_grad=True, device='cuda')

    def forward(self, x: torch.Tensor, mask: torch.Tensor, with_weights=False):
        # Give the constant
        B, N, D = x.size()
        x = super().forward(x, mask)
        C = self.num_anchor
        num_valid = mask.view(-1)
        # Calculate the score of each video
        dist = torch.norm(x.view(B, N, 1, D) - self.anchors, p=2, dim=3).view(B, N, C)  # B * N * C
        score = torch.zeros((B, N, C), dtype=torch.float, device='cuda')
        for i in range(B):
            score[i, :num_valid[i], :] = self.sparse_max(1 / dist[i, :num_valid[i], :])

        # Give weights along the N dim
        attribute = torch.argmax(score, dim=2)  # B * N
        weights = torch.zeros((B, N), dtype=torch.float, device='cuda')
        for i in range(B):
            try:
                index = torch.bincount(attribute[i, :num_valid[i]]).argmax()
            except RuntimeError as e:
                print(e)
                print(num_valid[i])
                index = 0
            weights[i] = score[i, :, index]

        feature = (x * weights.view(B, N, 1)).sum(1)

        if self.training and self.regular > 0:
            loss = 0
            if self.regular % 2 == 1:
                loss += 50 * dist[:, attribute.detach()].mean() / D

            if self.regular % 2 == 0:
                center = self.anchors.mean(0)
                loss += -torch.norm(self.anchors - center.view(1, -1), p=2, dim=1).mean() / D
            self.local_loss = loss

        if with_weights:
            return feature, weights
        else:
            return feature


class SEAttention(nn.Module):

    def __init__(self, in_channel, reduction):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduction, in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean(2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return (module_input * x.view(-1, 1)).sum(1)


class NeXtVLAD(BaseAggregator):

    def __init__(self, in_channels=512, num_cluster=4, lmd=2, g=8):

        super().__init__(in_channels, (lmd * in_channels) // g)
        self.nc = num_cluster
        self.G = g
        self.lmd = lmd
        self.centroid = nn.Parameter(torch.rand(num_cluster, (lmd * in_channels) // g, dtype=torch.float),
                                     requires_grad=True)
        self.expand = nn.Linear(in_channels, lmd * in_channels)
        self.alpha1 = nn.Linear(lmd * in_channels, g * num_cluster)
        self.alpha2 = nn.Sequential(nn.Linear(lmd * in_channels, g),
                                    nn.Softmax(2))

    def forward(self, x, mask):
        b, m, d = x.size()

        x = super().forward(x, mask)

        new_dims = (self.lmd * d) // self.G
        x = self.expand(x)  # x: b * m * lmd*d
        a1 = self.alpha1(x)  # a1: b * m * g*nc
        a1 = a1.view(b, m, self.G, self.nc)  # a1: b * m * g * nc
        a1 = F.softmax(a1, 3)  # a1: b * m * g * nc
        a2 = self.alpha2(x)  # a2: b * m * g
        x = x.view(b, m, self.G, 1, new_dims) - self.centroid  # x: b * m * g * nc * nd
        x = x * a1.view(b, m, self.G, self.nc, 1)
        x = (x * a2.view(b, m, self.G, 1, 1)).sum((2, 3)).mean(1)
        x = F.normalize(x, p=2, dim=1)
        return x


class NetVLAD(BaseAggregator):
    """NetVLAD layer implementation"""

    def __init__(self, in_channels=512, num_cluster=8, reduction=8, alpha=100.0):
        """
        Args:
            num_cluster : int
                The number of clusters
            in_channels : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
        """
        reduced_dims = in_channels // reduction
        super().__init__(in_channels, num_cluster * reduced_dims)
        self.num_clusters = num_cluster
        self.dim = in_channels
        self.alpha = alpha
        self.reduction = nn.Linear(in_channels, reduced_dims, bias=True)
        self.linear = nn.Linear(reduced_dims, num_cluster, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_cluster, reduced_dims))
        # self._init_params()

    def _init_params(self):
        self.linear.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.linear.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x, mask):
        x = self.reduction(x)

        b, m, d = x.size()  # x: (batch, length, dims)

        x = super().forward(x, mask)
        # soft-assignment
        soft_assign = self.linear(x)  # soft: b, m, n
        soft_assign = F.softmax(soft_assign, dim=2)

        # calculate residuals to each clusters
        residual = x.view(b, m, 1, d) - self.centroids.view(1, 1, self.num_clusters, d)
        # residual: b * m * n * d
        residual *= soft_assign.unsqueeze(3)
        vlad = residual.sum(dim=1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(b, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class Attention(BaseAggregator):
    def __init__(self, in_channels: int):
        super().__init__(in_channels, in_channels)
        self.attention = nn.Sequential(nn.Linear(in_channels, 1),
                                       nn.ReLU(inplace=True),
                                       SparseMax(1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = super().forward(x, mask)

        weights = self.attention(x)
        out = (x * weights).sum(1)

        return out


class Average(BaseAggregator):
    def __init__(self, in_channels: int):
        super().__init__(in_channels, in_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = super().forward(x, mask)
        return x.sum(1) / mask
