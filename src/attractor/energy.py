import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
# from discreteMarkovChain import markovChain
from src.decode.bump import decode_bump, circcvl
from src.preprocess.helpers import standard_scaler_BL, preprocess_X


def replace_zero_cols(matrix):
    nrows, ncols = matrix.shape
    for col in range(ncols):
        if np.all(matrix[:, col] == 0):  # Check for zero column
            # Find the non-zero neighbor columns indices to the left and right
            left = col - 1
            while left >= 0 and np.all(matrix[:, left] == 0):
                left -= 1
            right = col + 1
            while right < ncols and np.all(matrix[:, right] == 0):
                right += 1

            if left >= 0 and right < ncols:
                # Both neighbors found, take the average
                avg_col = (matrix[:, left] + matrix[:, right]) / 2
            elif left >= 0:
                # Only left neighbor found
                avg_col = matrix[:, left]
            elif right < ncols:
                # Only right neighbor found
                avg_col = matrix[:, right]
            else:
                raise ValueError("No non-zero neighbors found for column {0}".format(col))

            # Replace the zero column with the average of the neighbors
            matrix[:, col] = avg_col

    return matrix


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
    bins = np.linspace(0, 2 * np.pi, num_bins-1, endpoint=False)
    # bins = np.linspace(phase.min(), phase.max(), num_bins-1, endpoint=True)

    X_discrete = np.digitize(phase, bins)

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

    # bins = np.linspace(-np.pi, np.pi, num_bins-1, endpoint=False)
    bins = np.linspace(0, 2.0 * np.pi, num_bins-1, endpoint=False)
    if verbose:
        print('bins', bins)

    X_discrete = np.digitize(phase, bins)
    if verbose:
        print('X_bins', X_discrete.shape)

    # Initialize transition matrix
    matrix = np.ones((num_bins, num_bins))

    # Compute transitions
    for i in range(X_discrete.shape[0]): # trials
        for j in range(X_discrete.shape[1] - 1): # bins
            matrix[X_discrete[i, j], X_discrete[i, j+1]] += 1
            # matrix[X_discrete[i, j+1], X_discrete[i, j]] += 1

    # matrix = circcvl(matrix, windowSize=int(0.1*num_bins), axis=0)

    # Normalize transition matrix (to make it stochastic)
    # col_sum = matrix.sum(axis=1)

    # while np.any(col_sum==0):
    #     for i in range(matrix.shape[0]):
    #         if col_sum[i] == 0:
    #             if i == matrix.shape[0]-1:
    #                 if col_sum[i-1]>0 and col_sum[0]>0:
    #                     matrix[i] = (matrix[i-1] + matrix[0]) / 2.0
    #             elif i<matrix.shape[0]-1:
    #                 if col_sum[i-1]>0 and col_sum[i+1]>0:
    #                     matrix[i] = (matrix[i-1] + matrix[i+1]) / 2.0

    #     col_sum = matrix.sum(axis=1)

    # matrix = replace_zero_cols(matrix)

    col_sum = matrix.sum(axis=1)
    matrix = matrix / col_sum[:, np.newaxis]

    # row_sum = matrix.sum(axis=0)
    # print('row_sum', np.where(row_sum==0)[0], row_sum[0])
    # col_sum = matrix.sum(axis=1)
    # print('col_sum', np.where(col_sum==0)[0], col_sum[0])

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

def compute_energy_landscape(steady_state, window):
    # Compute the energy landscape as the negative log of the steady state distribution
    Z = np.sum(steady_state)
    energy = -np.log(steady_state) + np.log(Z)

    # energy = np.zeros(steady_state.shape) * np.nan
    # for i in range(steady_state.shape[0]):
    #     energy[i] = np.log(Z)
    #     if steady_state[i] > 0:
    #         energy[i] += -np.log(steady_state[i])

    # windowSize = int(window * energy.shape[0])
    # energy = circcvl(energy, windowSize=windowSize)

    Emin = np.nanmin(energy)
    energy = (energy - Emin)
    denom = np.nansum(energy)
    energy = energy / denom

    # denom = np.nansum(energy)
    # energy = energy / denom
    # Emin = np.nanmin(energy)
    # energy = energy - Emin

    return energy



def run_energy(X_, num_bins, bins, task, window, IF_HMM=0, VERBOSE=0, n_iter=100, IF_SYNT=0, bias=0.5, IF_NORM=0):

    if task=='all':
      X = np.vstack(X_)
    elif task==13:
      X = np.vstack((X_[0], X_[-1]))
    else:
      X = X_[task]

    print('X', X.shape)
    if IF_NORM:
        X = preprocess_X(X, scaler="robust", avg_noise=0, unit_var=0)

    if bins is not None:
      X = X[..., bins]

    _, phase = decode_bump(X, axis=1)
    # phase += np.pi

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

    energy = compute_energy_landscape(steady_state, window)

    if VERBOSE:
        print('energy', energy.shape)
        print(energy)

    return energy


def plot_energy(energy, ci=None, window=.9, ax=None, SMOOTH=0, color='r'):
    if ax is None:
        fig, ax = plt.subplots()

    theta = np.linspace(0, 360, energy.shape[0], endpoint=False)
    energy = energy[1:]
    theta = theta[1:]

    windowSize = int(window * energy.shape[0])
    if SMOOTH:
        energy = circcvl(energy, windowSize=windowSize)

    # theta = np.linspace(-180, 180, energy.shape[0], endpoint=False)

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
