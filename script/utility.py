import numpy as np
import scipy.sparse as sp
import torch

def calc_gso(dir_adj, gso_type):
    if sp.issparse(dir_adj):
        id = sp.identity(dir_adj.shape[0], format='csc')
        # Symmetrizing an adjacency matrix
        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
        #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
        if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
            adj = adj + id
    
        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
            or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj
        
        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = sp.diags(row_sum_inv, format='csc')
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')
    
    else:
        id = np.identity(dir_adj.shape[0])
        # Symmetrizing an adjacency matrix
        adj = np.maximum(dir_adj, dir_adj.T)
        #adj = 0.5 * (dir_adj + dir_adj.T)

        if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
            adj = adj + id

        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
            or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = np.sum(adj, axis=1)
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(row_sum_inv_sqrt)
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj

        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = np.sum(adj, axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = np.diag(row_sum_inv)
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_sgc_feature(gso, feature, K, device):
    if device == torch.device('cpu'):
        for k in range(K):
            feature = gso.dot(feature)
    else:
        for k in range(K):
            if gso.is_sparse:
                feature = torch.sparse.mm(gso, feature)
            else:
                feature = torch.mm(gso, feature)
    
    return feature

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.complex64 or sp_mat.dtype == np.complex128:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, device=device, requires_grad=False)
    elif sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def calc_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    accuracy = correct / len(labels)

    return accuracy