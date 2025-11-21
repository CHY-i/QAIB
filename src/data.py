import stim
import torch
from torch.utils.data import TensorDataset


def generate_data(d, r, num_shots, cir_type, error_prob=0.001, seed=85173):
    #['simulation', 'baqis', '...']
    if cir_type == 'sc':
        circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                        rounds=r,
                                        distance=d,
                                        after_clifford_depolarization=error_prob,
                                        after_reset_flip_probability=error_prob,
                                        before_measure_flip_probability=error_prob,
                                        before_round_data_depolarization=error_prob)
        
        dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
        sampler = dem.compile_sampler(seed=seed)
        dets, obvs, _ = sampler.sample(shots=num_shots)

    elif cir_type == 'bb':
        None
    
    dets, obvs = torch.from_numpy(dets*1.)
    dataset = TensorDataset(dets, obvs)
    return  dataset