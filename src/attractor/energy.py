import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
# from discreteMarkovChain import markovChain
from src.decode.bump import decode_bump, circcvl
from src.preprocess.helpers import standard_scaler_BL, preprocess_X


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def biased_markov(bias, num_bins):

    N = num_bins  # Number of states
    states = np.linspace(0, 360, N)  # State values from 0 to 360
    attractors = [0, 90]  # Positions of the attractors
    width = 30  # Determines how 'wide' the attraction is
    depth = 10   # Determines how 'deep' the attraction is

    # Create a 'potential' or 'energy' landscape with minima at the attractors
    potential = np.zeros_like(states)
    for a in attractors:
        potential += depth * (1 - gaussian(states, a, width))

    # Convert the potential into transition probabilities using Boltzmann factors
    prob = np.exp(-potential)
    prob /= np.sum(prob)  # Normalize

    # Package the probabilities into a transition matrix
    trans_matrix = np.zeros((N, N))
    for i in range(N):
        trans_matrix[i, :] = np.roll(prob, i)

    trans_matrix /= np.sum(trans_matrix, axis=1)[:, None]  # Normalize rows

    return trans_matrix, prob

def compute_transition_matrix_hmm(phase, num_bins, n_iter=100):

    # Create a Gaussian HMM
    model = hmm.GaussianHMM(n_components=num_bins, covariance_type="full", n_iter=n_iter)
    # model = hmm.PoissonHMM(n_components=num_bins, n_iter=n_iter)

    # Estimate model parameters.
    # Note: you would need to reshape your observations to be 2D array
    bins = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    # bins = np.linspace(phase.min(), phase.max(), num_bins-1, endpoint=True)

    # X_discrete = np.digitize(phase, bins)
    X_discrete = np.digitize(X, bins, right=False)-1

    # lgt = X_discrete.shape[-1]
    # length = np.ones(X_discrete.shape[0]) * lgt

    X_discrete = X_discrete.reshape(-1, 1)
    model.fit(X_discrete)

    # Transition probabilities between hidden states
    transition_matrix = model.transmat_
    # print('matrix', transition_matrix.shape)

    return transition_matrix


def compute_transition_matrix(phase, num_bins, verbose=0):

    if verbose:
        print('phase', phase.shape, phase.min() * 180 / np.pi, phase.max() * 180 / np.pi)

    bins = np.linspace(0, 2.0 * np.pi, num_bins, endpoint=False)
    # bins = np.linspace(-np.pi, np.pi, num_bins-1, endpoint=False)

    if verbose:
        print('bins', bins)

    X_discrete = np.digitize(phase, bins) - 1

    if verbose:
        print('X_bins', X_discrete.shape)

    # Initialize transition matrix
    matrix = np.ones((num_bins, num_bins))

    # Compute transitions
    for i in range(X_discrete.shape[0]): # trials
        for j in range(X_discrete.shape[1] - 1): # bins
            if (X_discrete[i,j]<num_bins) and (X_discrete[i,j+1]<num_bins):
                matrix[X_discrete[i, j], X_discrete[i, j+1]] += 1

    col_sum = matrix.sum(axis=1)

    # Avoid division by zero by setting denominators to 1 where there are no transitions
    # col_sum[col_sum == 0] = 1

    matrix = matrix / col_sum[:, np.newaxis]
    # matrix[:, 0] = matrix[:, -1]

    return matrix

def compute_steady_state(p_transition, VERBOSE=0):
    """This implementation comes from Colin Carroll, who kindly reviewed the notebook"""
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )

    # Moore-Penrose pseudoinverse = (A^TA)^{-1}A^T
    pinv = np.linalg.pinv(A)
    # Return last row
    return pinv.T[-1]

def compute_energy_landscape(steady_state):
    # Compute the energy landscape as the negative log of the steady state distribution
    Z = np.sum(steady_state)
    energy = -np.log(steady_state) + np.log(Z)

    Emin = np.nanmin(energy)
    energy = (energy - Emin)
    denom = np.nansum(energy)
    energy = energy / denom

    return energy

def run_energy(X_, num_bins, bins, bins0, task, window, IF_HMM=0, VERBOSE=0, n_iter=100, IF_SYNT=0, bias=0.5, IF_NORM=0):

    # X_[1][:][~bins0] = np.nan
    # X_[2][:][~bins0] = np.nan

    if task=='all':
      X = np.vstack(X_)
    else:
      X = X_[task]

    # print('X', X.shape)
    if IF_NORM:
        X = preprocess_X(X, scaler="robust", avg_noise=0, unit_var=0)

    if bins is not None:
      X = X[..., bins]

    try:
        _, phase = decode_bump(X, axis=1)
    except:
        phase = X

    if IF_HMM:
        transition_matrix = compute_transition_matrix_hmm(phase, num_bins=num_bins, n_iter=n_iter)
    elif IF_SYNT:
        transition_matrix, prob = biased_markov(bias=bias, num_bins=num_bins)
    else:
        transition_matrix = compute_transition_matrix(phase, num_bins=num_bins, verbose=VERBOSE)

    if VERBOSE:
        print('transition mat', transition_matrix.shape)
        print(transition_matrix)

    steady_state = compute_steady_state(transition_matrix, VERBOSE)

    # mc = markovChain(transition_matrix)
    # mc.computePi('power') #We can also use 'power', 'krylov' or 'eigen'
    # steady_state = mc.pi

    if VERBOSE:
        print('steady state', steady_state.shape)
        print(steady_state)

    energy = compute_energy_landscape(steady_state)

    if VERBOSE:
        print('energy', energy.shape)
        print(energy)

    return energy


def plot_energy(energy, ci=None, window=.9, ax=None, SMOOTH=0, color='r'):
    if ax is None:
        fig, ax = plt.subplots()

    theta = np.linspace(0, 360, energy.shape[0], endpoint=False)

    windowSize = int(window * energy.shape[0])
    if SMOOTH:
        energy = circcvl(energy, windowSize=windowSize)

    ax.plot(theta, energy * 100, lw=4, color=color)

    if ci is not None:
        ax.fill_between(
            theta,
            (energy - ci[:, 0]) * 100,
            (energy + ci[:, 1]) * 100,
            alpha=0.1, color=color
        )

    ax.set_ylabel('Energy')
    ax.set_xlabel('Pref. Location (°)')
    ax.set_xticks([0, 90, 180, 270, 360])
    # ax.set_xticks([-180, -90, 0, 90, 180])
    # plt.ylim([0, 2])
    # plt.savefig('landscape_' + mouse + '.svg', dpi=300)
