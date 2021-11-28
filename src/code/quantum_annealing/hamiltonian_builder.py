from pyqubo import Binary, Constraint, Array


class HamiltonianBuilder(object):
    def __init__(self, D_matrix, Q_array=None):
        self.n = D_matrix.shape[0]
        self.D_matrix = D_matrix
        self.Q_array = Q_array
        self.x = Array.create('x', (self.n, self.n), 'BINARY')

    def __build_H_A(self):
        H_a1 = 0
        for v in range(self.n):
            H_temp = 1
            for j in range(self.n):
                H_temp -= self.x[v, j]

            H_a1 += H_temp * H_temp

        H_a2 = 0
        for j in range(self.n):
            H_temp = 1
            for v in range(self.n):
                H_temp -= self.x[v, j]

            H_a2 += H_temp * H_temp

        H_a3 = 0
        for u in range(self.n):
            for v in range(self.n):
                k = 1
                for j in range(self.n):
                    H_a3 += self.x[u, j] * self.x[v, k]

                    k += 1
                    if k == self.n:
                        k = 0

        # remove the comment to add the contribute of H_a3 in case the tsp is not fully connected
        return H_a1 + H_a2  # + H_a3

    def __build_H_B(self):
        H_b = 0
        for u in range(self.n):
            for v in range(self.n):
                k = 1
                for j in range(self.n):
                    H_b += self.D_matrix[u, v] * self.x[u, j] * self.x[v, k]

                    k += 1
                    if k == self.n:
                        k = 0

        return H_b

    def __build_H_C(self):
        if self.Q_array is None:
            raise Exception('Hamiltonian built without Saturation modeling, Hamiltonian H_C not present')

        H_c = 0
        for u in range(self.n):
            for v in range(self.n):
                k = 1
                for j in range(self.n - 1):
                    H_c += self.x[u, j] * self.x[v, k] * ((self.Q_array[v] - self.Q_array[u]) ** 2)

                    k += 1
                    if k == self.n: k = 0

        return H_c

    def get_hamiltonian(self, norm_dict):
        A = norm_dict['A_Normalization']
        B = norm_dict['B_Normalization']
        C = norm_dict['C_Normalization']

        # build hamiltonians for tsp, H_A is always present for the Hamiltonian circuit definition
        H_A = A * self.__build_H_A()
        H = H_A

        # build hamiltonian for modeling durations, in the general case it is always needed
        # fort testing purpose if D_Matrix is None we build H only with H_A
        if self.D_matrix is not None:
            H_B = B * self.__build_H_B()
            H += H_B

        # build hamiltonian for modeling saturations, in the general case it is always needed
        # for testing purpose if Q_Array is None we build H only with H_A
        if self.Q_array is not None:
            H_C = C * self.__build_H_C()
            H += H_C

        return H
