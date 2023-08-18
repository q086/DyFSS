import os
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import random
import torch.optim as optim
import torch.nn.functional as F
from cluster_alignment import cluster_alignment_simple
from metric import cluster_accuracy
from losses import pq_loss_func, dist_2_label, sim_loss_func
from utils import get_dataset, weights_init, sparse_to_tuple
from vgae import VGAE, Discriminator
from MoeSSL import MoeSSL
import scipy.sparse as sp


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', help='use cpu or gpu.')
parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='photo', help='Type of dataset.')
parser.add_argument('--encoder', type=str, default="ARVGA", help='the model name')
parser.add_argument('--hid_dim', type=int, nargs='+', default=[256, 128, 512])
parser.add_argument('--lr_pretrain', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--pretrain_epochs', type=int, default=400, help='Number of pretrained totalssl epochs.')
parser.add_argument('--w_ssl_stage_one', type=float, default=0.25, help='the ssl_loss weight of pretrain stage one')
parser.add_argument('--use_ckpt', type=int, default=1, help='whether to use checkpoint, 0/1.')
parser.add_argument('--save_ckpt', type=int, default=1, help='whether to save checkpoint, 0/1.')

parser.add_argument('--top_k', type=int, default=5, help='The number of experts to choose.')
parser.add_argument('--st_per_p', type=float, default=0.5, help='The threshold of the pseudo positive labels.')
parser.add_argument('--lr_train_fusion', type=float, default=0.001, help='train pseudo labels learning rate.')
parser.add_argument('--labels_epochs', type=int, default=250, help='Number of epochs to train.')

parser.add_argument('--w_pq', type=float, default=1, help='weight of loss pq.')
parser.add_argument('--w_ssl_stage_two', type=float, default=0.01, help='the ssl_loss weight of train stage')
args = parser.parse_args()
opt = vars(args)


def main():
    set_seed(args.seed)
    print(args)
    data = get_dataset(args.dataset, True)
    feat_dim = data.features.shape[1]
    cluster_num = data.labels.max().item() + 1
    n_nodes = data.features.shape[0]

    encoder = VGAE(feat_dim, args.hid_dim[0], args.hid_dim[1], 0.0, cluster_num)
    modeldis = Discriminator(args.hid_dim[0], args.hid_dim[1], args.hid_dim[2])
    modeldis.apply(weights_init)

    set_of_ssl = ['PairwiseAttrSim', 'PairwiseDistance', 'Par', 'Clu', 'DGI']
    if data.adj.shape[0] > 5000:
        print("use the DGISample")
        local_set_of_ssl = [ssl if ssl != 'DGI' else 'DGISample' for ssl in set_of_ssl]
    else:
        local_set_of_ssl = set_of_ssl
    model = MoeSSL(data, encoder, modeldis, local_set_of_ssl, args, device=args.device).to(args.device)

    if args.use_ckpt == 1:
        print("load the ssl train ARVGA model")
        model.encoder.load_state_dict(
            torch.load(
                f'./pretrained/{args.encoder}_{args.dataset}_{args.hid_dim[0]}.pkl'))  # load ARVGA_dataset.pkl
    else:
        model.train()
        model.TotalSSLpretrain()

    model.encoder.eval()
    bottom_embeddings, _, mu_z, _, _ = model.encoder(model.processed_data.features, model.processed_data.adj_norm)
    acc, nmi, f1, mu_pred, kmeans = model.evaluate_pretrained(mu_z)
    print('Z-embddings clustering result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
    z_embeddings = mu_z.detach()

    # get the pseudo labels
    cluster_center_z = torch.from_numpy(kmeans.cluster_centers_)
    selected_idx_z, pred_labels_z = generate_pseudo_labels_percentile(z_embeddings, cluster_center_z,
                                                                      data, cluster_num, args.st_per_p)

    params = list(model.gate.parameters())  # gate
    # params = []
    for ix, agent in enumerate(model.ssl_agent):  # five ssl_agent
        if agent.disc2 is not None:
            params = params + list(agent.disc2.parameters())
        if hasattr(agent, 'gcn2'):
            params = params + list(agent.gcn2.parameters())

    def get_adj_label(adj):
        sim_matrix = adj.tocsr()
        sim_label = sim_matrix + sp.eye(sim_matrix.shape[0])
        sim_label = sparse_to_tuple(sim_label)
        sim_label = torch.sparse.FloatTensor(torch.LongTensor(sim_label[0].T), torch.FloatTensor(sim_label[1]),
                                             torch.Size(sim_label[2]))
        return sim_label

    adj_label = get_adj_label(data.adj)

    optimizer_train = None
    cluster_centers = None
    # Stage 2: training
    for epoch in range(args.labels_epochs):
        print(epoch)
        model.train()
        fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
        if epoch == 0:
            acc, nmi, f1, moe_pred_labels, kmeans = model.evaluate_pretrained(fusion_emb)
            print('fusion_emb cluster result: {:.2f}  {:.2f}  {:.2f}'.format(acc * 100, nmi * 100, f1 * 100))
            tenor_pseudo_labels = pred_labels_z.unsqueeze(1).cpu()
            pseudo_labels_onehot = torch.zeros(n_nodes, cluster_num).scatter_(1, tenor_pseudo_labels, 1)

            # labels cluster_alignment, return to get the aligned labels
            align_labels_z = torch.from_numpy(
                cluster_alignment_simple(data, moe_pred_labels, [pseudo_labels_onehot])[0])
            cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cuda(),
                                       requires_grad=True)
            params = params + [cluster_centers]
            optimizer_train = optim.Adam(params, lr=args.lr_train_fusion, weight_decay=0.0)

        optimizer_train.zero_grad()

        adj_sim_pred = sample_sim(fusion_emb)
        sim_loss = sim_loss_func(adj_sim_pred.view(-1), adj_label.to(args.device).to_dense().view(-1))

        losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
        sup_loss = F.nll_loss(torch.log(q[selected_idx_z]), align_labels_z[torch.LongTensor(selected_idx_z)].to(args.device))
        ssl_loss = model.get_ssl_loss_stage_two(z_embeddings, model.processed_data.adj_norm)

        loss = sup_loss + sim_loss + args.w_pq * losspq + args.w_ssl_stage_two * ssl_loss
        print(
            "loss pq: {}, pos_loss: {}, ssl_loss: {}, sim_loss: {}".format(losspq, sup_loss, ssl_loss, sim_loss,))

        loss.backward()
        optimizer_train.step()

        model.eval()
        fusion_emb, _, _ = model.FeatureFusionForward(z_embeddings)
        losspq, p, q = pq_loss_func(fusion_emb, cluster_centers)
        cluster_pred_score, cluster_pred = dist_2_label(q)
        acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), data.labels, cluster_num)
        print('Evaluate encoder_decoder clustering result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

    print("training ending...")
    model.eval()
    fusion_emb, _, nodes_weight = model.FeatureFusionForward(z_embeddings)
    _, p, q = pq_loss_func(fusion_emb, cluster_centers)
    _, cluster_pred = dist_2_label(q)
    acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), data.labels, cluster_num)
    print('Evaluate encoder_decoder clustering result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))


def init():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print("-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")
    print("-----------------------------------------------------------------")


# def generate_centers(cluster_centers, emb_unconf, y_pred):
#     nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb_unconf.cpu().detach().numpy())
#     # nn.kneighbors(): 得到上述emb_unconf中分别距离cluster_centers最近的点，这里体现为indices, 是上述点的index
#     _, indices = nn.kneighbors(cluster_centers.detach().numpy())
#     return indices[y_pred]  # 再用y_pred一套，那么就得到y_pred 的标签0-k,变成了对应的index

#
# def similarity_inds(z, upper_threshold):
#     f_adj = np.matmul(z, np.transpose(z))
#     cosine = f_adj
#     cosine = cosine.reshape([-1, ])  # len(consine表示所有节点的个数，n_nodes)
#     pos_num = round(upper_threshold * len(cosine))  # positive samples 样本个数，随着训练过程，逐渐变小
#     neg_num = pos_num
#
#     pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]  # 默认是从小到大排序
#     neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
#     print("pos_inds_length:", len(pos_inds))
#     print("neg_inds_length:", len(neg_inds))
#
#     return np.array(pos_inds), np.array(neg_inds)

#
# def get_sim_label(adj, pseudo_labels_z, emb, unconf_indices, cluster_centers, pos_inds, neg_inds):
#
#     n_nodes = emb.shape[0]
#
#     # 如果选择计算得到的相似性矩阵作为基础，（把所有的pos都看作1，然后把处于不同聚簇的边一定要断开，就是先用pos和neg构成一个简单的相似度矩阵）
#     # 得到对应的xind, yind
#     # sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=len(pos_inds)))
#     # sampled_neg = torch.LongTensor(neg_inds)
#     # sampled_inds = torch.cat((torch.LongTensor(pos_inds), sampled_neg), 0)
#     # xind_pos = torch.LongTensor(pos_inds) // n_nodes
#     # yind_pos = torch.LongTensor(pos_inds) % n_nodes
#     # xind_neg = torch.LongTensor(neg_inds) // n_nodes
#     # yind_neg = torch.LongTensor(neg_inds) % n_nodes
#     #
#     sim_matrix = adj.tolil()   # 基于邻接矩阵进行修改
#     # for i in range(len(pos_inds)):   # 邻接矩阵对应位置为1
#     #     sim_matrix[xind_pos[i], yind_pos[i]] = 1
#     # for i in range(len(neg_inds)):   # 邻接矩阵对应位置为0
#     #     sim_matrix[xind_neg[i], yind_neg[i]] = 0
#     # # 上面这部分，对于都有节点相似来说，保持原来节点之间的关系
#
#     # 下面这部分只是给可信集合的点，同一簇的节点连接起来（更紧密）
#     # return unconf_indices每一个节点中距离最近的unconf_indices索引  【把聚类中心换成距离最近的节点】
#     # unconf_indices 表示所有置信的节点集合
#
#     # if args.labels_cor == 1:
#     #     y_pred = pseudo_labels_z.cpu().numpy()  # all nodes pred labels
#     #     emb_unconf = emb[unconf_indices]
#     #     idx = unconf_indices[generate_centers(cluster_centers, emb_unconf, y_pred[unconf_indices])]
#     #     count_int = 0
#     #     count_out = 0
#     #     for i, k in enumerate(unconf_indices):  # 可信集合的每一个节点，，，# idx[i]表示距离最近的一个可信节点
#     #         indices_k = sim_matrix[k].tocsr().indices
#     #         if (not np.isin(idx[i], indices_k)) and (y_pred[k] == y_pred[idx[i]]):
#     #             sim_matrix[k, idx[i]] = 1
#     #             count_int = count_int + 1
#     #         for j in indices_k:
#     #             if np.isin(j, unconf_indices) and (y_pred[k] != y_pred[j]):
#     #                 sim_matrix[k, j] = 0
#     #                 count_out = count_out + 1
#     #
#     #     print("-----------count---------------")
#     #     print("count_int:", count_int)
#     #     print("count_out:", count_out)
#
#     sim_matrix = sim_matrix.tocsr()
#     # sim_matrix = adj.tocsr()
#     sim_label = sim_matrix + sp.eye(sim_matrix.shape[0])
#     sim_label = sparse_to_tuple(sim_label)
#     sim_label = torch.sparse.FloatTensor(torch.LongTensor(sim_label[0].T), torch.FloatTensor(sim_label[1]),
#                                          torch.Size(sim_label[2]))
#
#     return sim_label
#

# pred adj
def sample_sim(fusion_emb, xind=None, yind=None):

    def scale(z):
        zmax = z.max(dim=1, keepdim=True)[0]  # z的每一行的最大值
        zmin = z.min(dim=1, keepdim=True)[0]  # z的每一列的最小值
        z_std = (z - zmin) / (zmax - zmin)  # 将每一行的元素限制在0-1之间
        z_scaled = z_std
        return z_scaled
    fusion_scaled = scale(fusion_emb)  # 每一行限制在（0-1）之间
    fusion_norm = F.normalize(fusion_scaled)  # 每一行之和为1
    sim_adj = torch.mm(fusion_norm, fusion_norm.t())

    return sim_adj


def generate_pseudo_labels_percentile(mu_z, cluster_center_z, data, cluster_num, per):
    _, p, q_z = pq_loss_func(mu_z, cluster_center_z.cuda())
    cluster_pred_score, pred_labels_z = dist_2_label(q_z)
    tau_p = np.percentile(cluster_pred_score.detach().cpu().numpy(), per*100)
    selected_idx_z = cluster_pred_score >= tau_p
    selected_idx_z = [x for x, y in list(enumerate(selected_idx_z.tolist())) if y == True]
    selected_idx_z = np.asarray(selected_idx_z, dtype=int)
    print("The number of pseudo labels:", len(selected_idx_z))  # 1804 cons labels
    pesu_acc, pesu_nmi, pesu_f1 = cluster_accuracy(pred_labels_z[selected_idx_z].cpu(), data.labels[selected_idx_z],
                                                   cluster_num)
    print('Pesudo labels accuracy: {:.2f},{:.2f},{:.2f}'.format(pesu_acc * 100, pesu_nmi * 100, pesu_f1 * 100))
    print("-----------------------------------------------------------------------------")
    return selected_idx_z, pred_labels_z


def set_seed(seed):
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    init()
    main()