import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from hmmlearn import hmm

from dual_data.decode.bump import decode_bump, circcvl  

def compute_transition_matrix_hmm(phase, num_bins, n_iter=100):

    # Create a Gaussian HMM
    model = hmm.GaussianHMM(n_components=num_bins, covariance_type="full", n_iter=n_iter)
    # model = hmm.PoissonHMM(n_components=num_bins, n_iter=n_iter)
    
    # Estimate model parameters.
    # Note: you would need to reshape your observations to be 2D array
    X_discrete = np.digitize(phase, np.linspace(phase.min(), phase.max(), num_bins-1))

    X_discrete = X_discrete.reshape(-1, 1)
    model.fit(X_discrete)
    
    # Transition probabilities between hidden states
    transition_matrix = model.transmat_
    print('matrix', transition_matrix.shape)
    return transition_matrix, phase

def compute_transition_matrix(phase, num_bins):

    print('phase', phase.shape)
    X_discrete = np.digitize(phase, np.linspace(phase.min(), phase.max(), num_bins-1))
    print('bins', X_discrete.shape)
    # X_discrete = np.where(X_discrete==0, num_bins, X_discrete)
    
    # Initialize transition matrix
    matrix = np.zeros((num_bins, num_bins))
    
    # Compute transitions
    for i in range(X_discrete.shape[0]):
        for j in range(X_discrete.shape[1] - 1):
            matrix[X_discrete[i, j], X_discrete[i, j+1]] += 1 
    
    # Normalize transition matrix (to make it stochastic)
    denom = matrix.sum(axis=1).copy()
    denom[denom == 0.0] = 1.0
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if denom[j] == 0 :
                matrix[i, j] *= 0.0
            else:
                matrix[i, j] /= denom[j]
        
    return matrix, phase

def compute_steady_state(transition_matrix, VERBOSE=0):
    # The steady state distribution is the left eigenvector of the transition matrix corresponding to eigenvalue 1
    
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    
    if VERBOSE:
        print('eigenvalues', eigenvalues.shape)
        print(eigenvalues)

    if VERBOSE:
        print('eigenvectors', eigenvectors.shape)
        print(eigenvectors)
    
    steady_state = np.real(eigenvectors[:, np.isclose(eigenvalues, 1.0)])
    
    if VERBOSE:
        print('sum', steady_state.sum())
    
    steady_state /= steady_state.sum()  # Ensure the probabilities sum up to 1
    
    if VERBOSE:
        print(steady_state)    
    
    return steady_state.flatten()

def compute_energy_landscape(steady_state, window):
    # Compute the energy landscape as the negative log of the steady state distribution
    energy = np.zeros(steady_state.shape) * np.nan
    
    for i in range(steady_state.shape[0]):
        if steady_state[i] !=0 :
            energy[i] = -np.log(steady_state[i])

    # energy = circcvl(energy, windowSize=window)
    
    Emin = np.nanmin(energy)
    energy = energy - Emin
    
    denom = np.nansum(energy)
    energy = energy / denom
    # energy = (energy - np.nanmin(energy))/ denom
    
    # if energy[~np.isnan(energy)].size > 0:
    
    return energy


def run_energy(X, num_bins, window, IF_HMM=0, IF_PHASE=1, VERBOSE=0, n_iter=100):

    _, phase = decode_bump(X, axis=1)
    
    if IF_HMM:
        transition_matrix, phase = compute_transition_matrix_hmm(phase, num_bins=num_bins, n_iter=n_iter)
    else:
        transition_matrix, phase = compute_transition_matrix(phase, num_bins=num_bins)

    if VERBOSE:
        print('transition mat', transition_matrix.shape)
        print(transition_matrix)
    
    steady_state = compute_steady_state(transition_matrix, VERBOSE)
    
    if VERBOSE:
        print('steady state', steady_state.shape)
        print(steady_state)
    
    energy = compute_energy_landscape(steady_state, window)
    
    if VERBOSE:
        print('energy', energy.shape)
        print(energy)
        
    if IF_PHASE:
        return energy, phase
    else:
        return energy

def plot_energy(phase, energy, ci=None, window=.9, ax=None, SMOOTH=0):
    if ax is None:
        fig, ax = plt.subplots()

    window = int(window * energy.shape[0])
    if SMOOTH:
        energy = circcvl(energy, windowSize=window)
    
    theta = np.linspace(phase.min(), phase.max(), energy.shape[0]) * 180 / np.pi + 180
    ax.plot(theta, energy * 100, lw=4)
    
    if ci is not None:
        ax.fill_between(
            theta,
            (energy - ci[:, 0]) * 100,
            (energy + ci[:, 1]) * 100,
            alpha=0.2,
        )
    
    ax.set_ylabel('Energy (a.u.)')
    ax.set_xlabel('Pref. Location (°)')
    ax.set_xticks([0, 90, 180, 270, 360])
    # plt.ylim([0, 2])
    # plt.savefig('landscape_' + mouse + '.svg', dpi=300)
