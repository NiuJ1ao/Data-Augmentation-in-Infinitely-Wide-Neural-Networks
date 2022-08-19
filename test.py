import numpy as np

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

n = 60000
np.save("/vol/bitbucket/yn621/data/test_matrix", np.random.rand(n, n) + 1e-6*np.eye(n))

A, B, C, D = split(np.load("/vol/bitbucket/yn621/data/test_matrix.npy"), n//2, n//2)
np.save("/vol/bitbucket/yn621/data/A", A)
np.save("/vol/bitbucket/yn621/data/B", B)
np.save("/vol/bitbucket/yn621/data/C", C)
np.save("/vol/bitbucket/yn621/data/D", D)

def load_A():
    return np.load("/vol/bitbucket/yn621/data/A.npy")
def load_B():
    return np.load("/vol/bitbucket/yn621/data/B.npy")
def load_C():
    return np.load("/vol/bitbucket/yn621/data/C.npy")
def load_D():
    return np.load("/vol/bitbucket/yn621/data/D.npy")

D_inv = np.linalg.inv(load_D())
tmp = load_A() - load_B() @ D_inv @ load_C()
upperleft = np.linalg.inv(tmp)
upperright = -upperleft @ load_B() @ D_inv
lowerleft = -D_inv @ load_C() @ upperleft
lowerright = D_inv + D_inv @ load_C() @ upperleft @ load_B() @ D_inv
matrix_inverse = np.vstack([np.hstack([upperleft, upperright]), np.hstack([lowerleft, lowerright])])

a = np.load("/vol/bitbucket/yn621/data/test_matrix.npy")
# a_inv = np.load("/vol/bitbucket/yn621/data/test_matrix.npy")
print(np.allclose(a @ matrix_inverse, np.eye(n)))
print(a @ matrix_inverse)