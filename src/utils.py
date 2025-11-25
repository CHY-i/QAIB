import stim
import torch
import numpy as np

def build_spatial_mapping(circuit: stim.Circuit, padding=False, ):
    coords_dict = circuit.get_detector_coordinates()
    
    
    unique_xys = set()
    for idx, coords in coords_dict.items():
        if len(coords) >= 2:
            unique_xys.add((coords[0], coords[1]))
    sorted_spatial_locs = sorted(list(unique_xys), key=lambda p: (p[1], p[0]))
    if padding:
        n_pixels_row = max(loc[0] for loc in sorted_spatial_locs)//2 +1
        loc_to_idx = {loc: (loc[0]//2 + (loc[1]//2) * n_pixels_row) for i, loc in enumerate(sorted_spatial_locs)}
        num_spatial_features = n_pixels_row ** 2
    else:
        
        loc_to_idx = {loc: i for i, loc in enumerate(sorted_spatial_locs)}
        num_spatial_features = len(sorted_spatial_locs)
    

    num_detectors = circuit.num_detectors
    flat_to_round = torch.zeros(num_detectors, dtype=torch.long)
    flat_to_spatial = torch.zeros(num_detectors, dtype=torch.long)
    

    times = [coords_dict[i][2] if len(coords_dict[i]) > 2 else 0 for i in range(num_detectors)]
    unique_times = sorted(list(set(times)))
    time_to_round = {t: i for i, t in enumerate(unique_times)}
    
    for i in range(num_detectors):
        coords = coords_dict[i]
        x, y = coords[0], coords[1]
        t = coords[2] if len(coords) > 2 else 0
        flat_to_round[i] = time_to_round[t]
        flat_to_spatial[i] = loc_to_idx[(x, y)]+t*num_spatial_features
    return flat_to_round, flat_to_spatial


def SurfaceDataReshape(dets, num_pixels, rounds, flat_to_spatial):
    batch_size = dets.shape[0]
    spatial_dets = torch.zeros(batch_size, num_pixels*(rounds+1))
    spatial_dets[:, flat_to_spatial] = dets
    return spatial_dets.reshape(batch_size, rounds+1, -1)


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