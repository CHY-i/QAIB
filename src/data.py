import stim
import torch
from torch.utils.data import TensorDataset
from .qcc import qcc_circuit

def generate_sureface_data(d, r, num_shots, error_prob=0.001, seed=85173):
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


    dets, obvs = torch.from_numpy(dets*1.)
    dataset = TensorDataset(dets, obvs)
    return  dataset

def generate_bb_data(num_shots,
                    error_prob = 0.003, 
                    l = 5, 
                    m = 6, 
                    A_x_pows = [0], 
                    A_y_pows = [1,2], 
                    B_x_pows = [1,4], 
                    B_y_pows = [1], 
                    rounds = 2,
                    seed=85173):

    qccc = qcc_circuit(error_prob, l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, rounds)
    dem = qccc.detector_error_model(flatten_loops=True, decompose_errors=False)
    sampler = dem.compile_sampler(seed=seed)
    dets, obvs, _ = sampler.sample(shots=num_shots)
    dets, obvs = torch.from_numpy(dets*1.)
    dataset = TensorDataset(dets, obvs)
    return  dataset

def google_data():
    None