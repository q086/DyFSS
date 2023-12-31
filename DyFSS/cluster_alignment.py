import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def linear_sum_assign(bottom_label_one_hot, all_outputs):
    new_outs = []
    new_labels_list = []
    pvt_inds = bottom_label_one_hot.argmax(-1)
    for en in range(all_outputs.shape[0]):
        old_inds = all_outputs[en].argmax(-1)
        new_o = allignCLus(old_inds, pvt_inds, all_outputs[en])
        new_labels = new_o.argmax(-1)
        new_outs.append(new_o)
        new_labels_list.append(new_labels)
    ensemble_labels = np.array(new_outs).mean(0).argmax(-1)
    return new_labels_list, ensemble_labels


def allignCLus(cl1, cl2, cl_pr):
    y_true, y_pred = cl1, cl2
    Y_pred = y_pred
    Y = y_true
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
    DD = {}
    for i in range(len(ind[0])):
        DD[ind[1][i]] = ind[0][i]
    newCl = np.zeros_like(cl_pr)
    for i in range(cl_pr.shape[1]):
        newCl[:, DD[i]] = np.copy(cl_pr[:, i])
    return np.array(newCl)


def cluster_alignment_simple(data, bottom_pred, models_pred_labels_tensor_list):

    cluster_num = data.labels.max().item() + 1
    n_nodes = data.features.shape[0]
    tenor_pred_label = torch.LongTensor(bottom_pred).unsqueeze(1)
    bottom_label_one_hot = torch.zeros(n_nodes, cluster_num).scatter_(1, tenor_pred_label, 1)
    array_models_pred_labels = torch.stack(models_pred_labels_tensor_list).numpy()
    new_labels_list, _ = linear_sum_assign(bottom_label_one_hot.numpy(), array_models_pred_labels)

    return new_labels_list