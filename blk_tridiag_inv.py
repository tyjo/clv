"""
This was originally written for Theano. It has been modified for use
with numpy.
"""

"""
The MIT License (MIT)
Copyright (c) 2015 Evan Archer 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import scipy.linalg

def compute_blk_tridiag(AA, BB):
    '''
    Compute block tridiagonal terms of the inverse of a *symmetric* block tridiagonal matrix.
    
    All input & output assumed to be stacked numpy arrays. The function expects the off-diagonal
    blocks of the upper triangle & returns the upper triangle.

    Input: 
    AA - (T x n x n) diagonal blocks 
    BB - (T-1 x n x n) off-diagonal blocks (upper triangle)

    Output: 
    D  - (T x n x n) diagonal blocks of the inverse
    OD - (T-1 x n x n) off-diagonal blocks of the inverse (upper triangle)
    S  - (T-1 x n x n) intermediary matrix computation used in inversion algorithm 
 
    From: 
    Jain et al, 2006
    "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded Matrices"
    Note: Could be generalized to non-symmetric matrices, but it is not currently implemented.
    
    (c) Evan Archer, 2015
    '''
    if AA.shape[0] == 1:
        return np.linalg.pinv(AA[0]).reshape(AA.shape), np.zeros((0,0,0)), np.zeros((0,0,0))

    AA = np.copy(AA)
    BB = -np.copy(BB)

    d = AA.shape[1]
    nT = AA.shape[0]
    III = np.eye(d)

    # S intermediate matrix
    S = np.zeros(BB.shape)
    for idx in range(nT-2, -1, -1):
        if idx == nT - 2:
            S[idx] = np.dot(BB[-1], np.linalg.pinv(AA[-1]))
        else:
            S[idx] = np.dot(BB[idx],
                        np.linalg.pinv(
                            AA[idx+1] - np.dot(S[idx+1], np.transpose(BB[idx+1]) )))

    # Diagonal
    D = np.zeros(AA.shape)
    for idx in range(0, nT):
        if idx == 0:
            D[idx] = np.linalg.pinv(AA[idx] - np.dot(BB[idx], S[idx].T))
        elif idx < nT - 1:
            D[idx] = np.dot(np.linalg.pinv(AA[idx] - BB[idx].dot(S[idx].T)),
                            III + BB[idx-1].T.dot(D[idx-1]).dot(S[idx-1]))
        else:
            D[idx] = np.dot(np.linalg.pinv(AA[idx]),
                            III + BB[idx-1].T.dot(D[idx-1]).dot(S[idx-1]))

    # Upper off-diagonal
    OD = np.zeros(BB.shape)
    for idx in range(0, nT-1):
        OD[idx] = D[idx].dot(S[idx])

    return [D, OD, S]


def compute_blk_tridiag_inv_b(S,D,b):
    '''
    Solve Cx = b for x, where C is assumed to be *symmetric* block matrix.

    Input: 
    D  - (T x n x n) diagonal blocks of the inverse
    S  - (T-1 x n x n) intermediary matrix computation returned by  
         the function compute_sym_blk_tridiag

    Output: 
    x - (T x n) solution of Cx = b 

   From: 
    Jain et al, 2006
    "Numerically Stable Algorithms for Inversion of Block Tridiagonal and Banded Matrices"

    (c) Evan Archer, 2015
    '''
    b = np.array(b)
    if D.shape[0] == 1:
        return D[0].dot(b[0])

    nT = b.shape[0]
    d = b.shape[1]

    p = np.zeros(b.shape)
    p[-1] = b[-1]
    for idx in range(nT-2, -1, -1):
        p[idx] = b[idx] + S[idx].dot(p[idx+1])
    
    y = np.zeros(b.shape)
    q = np.zeros(b.shape)
    y[0] = D[0].dot(p[0])
    q[0] = S[0].T.dot(D[0].dot(b[0]))
    
    for idx in range(1, nT-1):
        #q[idx] = S[idx].T.dot( (q[idx-1] + D[idx].dot(b[idx].T).reshape(d)).T ).reshape(d)
        q[idx] = S[idx].T.dot( q[idx-1] + D[idx].dot(b[idx]) )
        y[idx] = D[idx].dot(p[idx]) + q[idx-1]

    y[nT-1] = D[nT-1].dot(p[nT-1]) + q[nT-2]

    return np.copy(y)

          
if __name__ == "__main__": 
    np.set_printoptions(suppress=True)
    # Build a block tridiagonal matrix 
    # npA = np.mat('1 6; 6 4')
    # npB = np.mat('2 7; 7 4')
    # npC = np.mat('3 9; 9 1')
    # npD = np.mat('7 2; 9 3')
    # npZ = np.mat('0 0; 0 0')

    npA = np.mat('1 1; 0 1')
    npB = np.mat('2 8; 3 2')
    npC = np.mat('5 9; 9 1')
    npD = np.mat('7 2; 8 3')
    npZ = np.mat('0 0; 0 0')

    # a 2x2 block tridiagonal matrix with 4x4 blocks
    fullmat = np.bmat([[npA,     npB, npZ,   npZ], 
                       [npB.T,   npC, npD,   npZ], 
                       [npZ,   npD.T, npC,   npB], 
                       [npZ,     npZ, npB.T, npC]])

    tA = npA
    tB = npB
    tC = npC
    tD = npD
    
    AAin = np.array([tA, tC, tC, tC])
    BBin = np.array([tB, tD, tB])

    D, OD, S = compute_blk_tridiag(AAin,BBin)

    print(D)
    print(OD)

    print(scipy.linalg.inv(fullmat)[:4,:4])

    # test solving the linear sysstem Ay=b
    # now let's implement the solver (IE, we want to solve for y in Ay=b)
    npb = np.asmatrix(np.arange(4*2).reshape((4,2)))
    b = npb

    y = compute_blk_tridiag_inv_b(S,D,b)
    print(y.reshape(8,1))

    print(np.linalg.pinv(fullmat).dot(npb.reshape(8,1)))

