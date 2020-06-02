import numpy as np

q = np.random.random((3, 3, 2))
print(q)
for row_num in range(3):
    for col_num in range(3):
        q[row_num, col_num, :] = q[row_num, col_num, :] / sum(q[row_num, col_num, :])
print(q)
