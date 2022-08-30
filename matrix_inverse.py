import jax.numpy as np
# import numpy as np

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def schur_inverse(matrix, recursion=1):
    A, B, C, D = split(matrix, matrix.shape[0]//2, matrix.shape[0]//2)
    if recursion <= 1:
        D_inv = np.linalg.inv(D)
        tmp = A - B @ D_inv @ C
        upperleft = np.linalg.inv(tmp)
        upperright = -upperleft @ B @ D_inv
        lowerleft = -D_inv @ C @ upperleft
        lowerright = D_inv + D_inv @ C @ upperleft @ B @ D_inv
        matrix_inverse = np.vstack([np.hstack([upperleft, upperright]), np.hstack([lowerleft, lowerright])])
        return matrix_inverse
    else:
        D_inv = schur_inverse(D, recursion-1)
        upperleft = schur_inverse(A - B @ D_inv @ C, recursion-1)
        upperright = -upperleft @ B @ D_inv
        lowerleft = -D_inv @ C @ upperleft
        lowerright = D_inv + D_inv @ C @ upperleft @ B @ D_inv
        matrix_inverse = np.vstack([np.hstack([upperleft, upperright]), np.hstack([lowerleft, lowerright])])
        return matrix_inverse

n = 60_000
a = np.arange(n * n).reshape((n, n)) + 1e-6 * np.eye(n)
# np.save("/vol/bitbucket/yn621/data/test_matrix", a)
print(a.shape)
# a_inv = np.linalg.inv(a)
# print(a_inv.shape)

# a = np.load("/vol/bitbucket/yn621/data/test_matrix.npy")
a_inv_ = schur_inverse(a)
print(np.allclose(a_inv_ @ a, np.eye(n)))