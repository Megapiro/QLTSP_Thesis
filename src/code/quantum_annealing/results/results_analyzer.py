import itertools


def qubits_number(embedded_graph):
    # return the number of qubits required in the embedding
    sublist = [values for keys, values in embedded_graph.items()]
    flat_list = set(itertools.chain(*sublist))

    return len(flat_list)


def chain_lengths(embedded_graph):
    # return the in and max chain length in the embedding provided
    max_chain_length = None
    min_chain_length = None

    for _, chain in embedded_graph.items():
        if max_chain_length is None:
            max_chain_length = len(chain)
            min_chain_length = len(chain)

        if len(chain) > max_chain_length:
            max_chain_length = len(chain)

        if len(chain) < min_chain_length:
            min_chain_length = len(chain)

    return min_chain_length, max_chain_length
