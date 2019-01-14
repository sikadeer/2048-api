import numpy as np

CAND = 16
map_table = { 2**i:i for i in range(1, CAND) }
map_table[0] = 0


def grid_ohe(arr):
    print(arr)
    ret = np.zeros(shape=(4, 4, CAND), dtype=float)
    for r in range(4):
        for c in range(4):
            ret[r, c, map_table[arr[r, c]]] = 1
    return ret
