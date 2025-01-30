import numpy as np
# from src.decode.bump import circcvl

class EnergyLandscape():

    def __init__(self, verbose=False):
        self.verbose = verbose

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

    def get_steady_state(self, p_transition):
        """This implementation comes from Colin Carroll"""

        n_states = p_transition.shape[0]

        A = np.vstack((p_transition.T - np.eye(n_states), np.ones(n_states)))

        # Moore-Penrose pseudoinverse = (A^TA)^{-1}A^T
        pinv = np.linalg.pinv(A)
        # Return last row
        return pinv.T[-1]

    def get_energy(self, steady_state, window=10):
        # Compute the energy landscape as the negative log of the steady state distribution (Boltzmann, kb=1)
        Z = np.sum(steady_state)
        energy = -np.log(steady_state) + np.log(Z)

        # energy = circcvl(energy, windowSize=window)

        Emin = np.nanmin(energy)
        energy = (energy - Emin)
        denom = np.nansum(energy)
        energy = energy / denom

        return energy

    def fit(self, X, bins, window=10):

        X_discrete = np.digitize(X, bins, right=False)-1
        num_bins = len(bins)

        self.transition_matrix = self.get_transition_matrix(X_discrete, num_bins=num_bins)
        if self.verbose:
            print('Transition matrix', self.transition_matrix.shape)

        self.steady_state = self.get_steady_state(self.transition_matrix)
        if self.verbose:
            print('Steady States', self.steady_state.shape)

        energy = self.get_energy(self.steady_state, window)
        if self.verbose:
            print('Energy', energy.shape)

        return energy
