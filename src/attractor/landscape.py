import numpy as np
from src.decode.bump import circcvl
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

class EnergyLandscape():

    def __init__(self, IF_HMM=0, verbose=False):
        self.IF_HMM = IF_HMM
        self.verbose = verbose


    def get_transition_hmm(self, X_discrete, num_bins, covariance_type='diag', n_iter=50):
        model = hmm.GaussianHMM(n_components=num_bins, covariance_type=covariance_type, n_iter=n_iter, init_params="")
        model.transmat_ = np.ones((num_bins, num_bins)) / num_bins

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_discrete)
        model.fit(X_scaled)

        return model.transmat_


    def get_transition_matrix(self, X_discrete, num_bins):

        # I assume at least 1 transition everywhere
        matrix = np.ones((num_bins, num_bins))

        for x in X_discrete:
            idx_pairs = zip(x[:-1], x[1:])
            for i, j in idx_pairs:
                matrix[i, j] += 1

        # # Compute transitions between t and t+1
        # for i in range(X_discrete.shape[0]): # trials
        #     for j in range(X_discrete.shape[1] - 1): # bins
        #         matrix[X_discrete[i, j], X_discrete[i, j+1]] += 1

        col_sum = matrix.sum(axis=1, keepdims=True)

        # Avoid division by zero by setting denominators to 1 where there are no transitions
        # col_sum[col_sum == 0] = 1

        matrix = matrix / col_sum

        return matrix

    def get_steady_state_old(self, p_transition):
        """This implementation comes from Colin Carroll"""

        n_states = p_transition.shape[0]

        A = np.vstack((p_transition.T - np.eye(n_states), np.ones(n_states)))

        # A = np.append(
        #     arr=p_transition.T - np.eye(n_states),
        #     values=np.ones(n_states).reshape(1, -1),
        #     axis=0
        # )

        # Moore-Penrose pseudoinverse = (A^TA)^{-1}A^T
        pinv = np.linalg.pinv(A)
        # Return last row
        return pinv.T[-1]


    def get_steady_state(self, p_transition):
        """Returns stationary distribution of the Markov chain."""
        eigvals, eigvecs = np.linalg.eig(p_transition.T)
        idx = np.argmin(np.abs(eigvals - 1))
        steady_state = np.real(eigvecs[:, idx])
        steady_state = steady_state / np.sum(steady_state)    # normalize
        steady_state[steady_state < 0] = 0                    # clip negatives if any due to numerics
        steady_state = steady_state / np.sum(steady_state)    # renormalize
        return steady_state


    def get_energy(self, steady_state, window=10):
        # Compute the energy landscape as the negative log of the steady state distribution (Boltzmann, kb=1)
        # Pi = exp(-Ei / kb / T) / Z where Z = sum(Pi)

        Z = np.sum(steady_state)
        energy = -np.log(steady_state) - np.log(Z)

        # energy = circcvl(energy, windowSize=window)

        Emin = np.nanmin(energy)
        energy = (energy - Emin)
        denom = np.nansum(energy)
        energy = energy / denom

        return energy

    def fit(self, X, bins, window=10, covariance_type='diag', n_iter=1000):

        num_bins = len(bins) - 1
        # X_discrete = np.digitize(X, bins, right=False)-1
        X_discrete = np.clip(np.digitize(X, bins)-1, 0, num_bins-1)

        if self.IF_HMM:
            self.transition_matrix = self.get_transition_hmm(X_discrete, num_bins=num_bins, covariance_type=covariance_type, n_iter=n_iter)
        else:
            self.transition_matrix = self.get_transition_matrix(X_discrete, num_bins=num_bins)

        if self.verbose:
            print('Transition matrix', self.transition_matrix.shape)

        self.steady_state = self.get_steady_state(self.transition_matrix)
        if self.verbose:
            print('Steady States', self.steady_state.shape)

        energy = self.get_energy(self.steady_state, window)

        if self.verbose:
            print('Energy', energy.shape)

        if window>0:
            energy = circcvl(energy, windowSize=window)

        return energy
