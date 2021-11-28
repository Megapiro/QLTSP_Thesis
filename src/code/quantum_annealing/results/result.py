import math


class Result(object):
    def print_response_data(self, do_print):
        n = 0
        pos_sets = []
        pos_energies = []
        response = self.response.aggregate().lowest()
        if do_print:
            # ------- Print results to user -------
            print('-' * 130)
            print('{:>45s}{:>60s}{:>22s}'.format('Set 1', 'Energy', "Count"))
            print('-' * 130)
        for sample, E, occ in response.data(fields=['sample', 'energy', "num_occurrences"]):
            bool_energy = False
            S0 = []
            S1 = []
            E1 = None
            for k, v in sample.items():
                if v == 0:
                    S0.append(k)
                else:
                    if not bool_energy:
                        E1 = E
                        bool_energy = True
                    S1.append(k)

            pos_sets.append(S1)
            pos_energies.append(E1)
            n = int(math.sqrt(len(S0) + len(S1)))
            if do_print:
                print('{:>30s}{:^30s}{:^15s}'.format(str(S1), str(E), str(occ)))

        return n, pos_sets, pos_energies

    def __init__(self, response, do_print):
        self.response = response
        self.n, self.pos_sets, self.pos_energies = self.print_response_data(do_print)

    @staticmethod
    def map_variables(pos_set):
        m_set = []
        for i in range(len(pos_set)):
            x = pos_set[i].replace(']', '').split('[')
            m_set.append([int(x[1]), int(x[2])])

        return m_set

    def split_solution(self):
        correct_solutions = []
        wrong_solutions = []

        for p_set in self.pos_sets:
            correct = True
            m_set = self.map_variables(p_set)
            s_res = sorted(m_set, key=lambda x: x[1])

            if len(s_res) != self.n:
                correct = False
            else:
                for i in range(0, int(len(s_res)), 2):
                    start_node = (s_res[i + 0])[0]
                    end_node = (s_res[i + 1])[0]

                    if abs(start_node - end_node) > 1:
                        correct = False

            if correct:
                correct_solutions.append(s_res)
            else:
                wrong_solutions.append(s_res)

        return correct_solutions, wrong_solutions

    def get_solution(self):
        correct_solutions, wrong_solutions = self.split_solution()

        print('-' * 130)
        print('{:>50s}{:>50s}'.format('Correct Solutions', 'Wrong Solutions'))
        print('-' * 130)
        dummy_print = '                                   '
        dummy_init_space = '                         '

        max_len = max(len(correct_solutions), len(wrong_solutions))
        for i in range(max_len):
            if i >= len(correct_solutions):
                print(dummy_print, end='                              |')
            else:
                s_res = correct_solutions[i]

                print(dummy_init_space, end='')
                for j in range(len(s_res) - 1):
                    print((s_res[j])[0], end='-->')

                print((s_res[len(s_res) - 1])[0], end='          |')

            if i >= len(wrong_solutions):
                print(dummy_print)
            else:
                s_res = wrong_solutions[i]

                print(dummy_init_space, end='')
                for j in range(len(s_res) - 1):
                    print((s_res[j])[0], end='-->')

                print((s_res[len(s_res) - 1])[0])

        return correct_solutions
