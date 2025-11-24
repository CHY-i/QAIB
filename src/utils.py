import torch


def surface_data_to_3D(x):
    '''convert surface data shape to [batch, d+1, d+1, round+1]'''
    
    None


def adj_matrix(pcm):
    return torch.sparse.mm(pcm, pcm.T)   

    
def PCM(dem, padding_nrow=0):
    
    pcm = torch.zeros([dem.num_detectors, dem.num_errors])
    l = torch.zeros([dem.num_observables, dem.num_errors])
    
   
    for i, e in enumerate(dem[:dem.num_errors]):
        Dec = e.targets_copy()

        for j in range(len(Dec)):
            D = str(Dec[j])
            if D.startswith('D'):
                idx = int(D[1:])
                pcm[idx, i] = 1.#e.args_copy()[0]

            if D.startswith('L'):
                idx = int(D[1:])
                l[idx, i] = 1
    if padding_nrow:
        padding_matrix = torch.zeros(padding_nrow, dem.num_errors)
        pcm = torch.vstack([padding_matrix, pcm, padding_matrix])
    return  pcm, l

def softmax_nonzero_rows(X):
    for i in range(X.shape[0]): 
        row = X[i, :]
        mask = row != 0     
        nonzero_values = row[mask]
        exp_values = torch.exp(nonzero_values)
        softmax_values = exp_values / torch.sum(exp_values) 
        X[i, mask] = softmax_values
    return X