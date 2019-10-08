import numpy as np
import sys

from scipy.special import logsumexp


def estimate_latent_from_observations(Y, pseudo_count=1e-5):
    X = []
    for y in Y:
        x = np.zeros((y.shape[0], y.shape[1]-1))
        for t in range(y.shape[0]):
            pt = y[t] / y[t].sum()
            pt = (pt + pseudo_count) / (pt + pseudo_count).sum()
            xt  = np.log(pt[:-1]) - np.log(pt[-1])
            x[t] = xt
        X.append(x)
    return X



def elastic_net(X, U, T, Q_inv, r_A, r_g, r_B, alpha=1, tol=1e-3, verbose=False, max_iter=100000):

    def gradient(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:,:yDim]
        g = AgB[:,yDim:(yDim+1)]
        B = AgB[:,(yDim+1):]
        grad[:,:yDim] += -2*alpha*(1-r_A)*A
        grad[:,yDim:(yDim+1)] += -2*alpha*(1-r_g)*g
        grad[:,(yDim+1):] += -2*alpha*(1-r_B)*B
        return -grad


    def generalized_gradient(AgB, grad, step):
            proximal = prv_AgB - step*grad

            # threshold A
            A = proximal[:,:yDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = proximal[:,yDim:(yDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = proximal[:,(yDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB_proximal = np.zeros(AgB.shape)
            AgB_proximal[:,:yDim] = A
            AgB_proximal[:,yDim:(yDim+1)] = g
            AgB_proximal[:,(yDim+1):] = B

            return (AgB - AgB_proximal)/step


    def objective(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*(f.dot(Q_inv)*f).sum()

        return -obj

    def stack_observations(X, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by yDim + 1 + uDim
        pgu_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                zt0 = np.concatenate((x[t-1], np.array([0])))
                pt0 = np.exp(zt0 - logsumexp(zt0))
                gt0 = np.ones(1)
                ut0 = u[t-1]
                pgu = np.concatenate((pt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t-1]
                    pgu_stacked = dt*pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))

        return x_stacked, pgu_stacked


    # assert 0 < r_A and r_A < 1
    # assert 0 < r_g and r_g < 1
    # assert 0 < r_B and r_B < 1
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, yDim + 1 + uDim ))

    x_stacked, pgu_stacked = stack_observations(X, U, T)
    prv_obj = np.inf
    obj = objective(AgB, x_stacked, pgu_stacked)

    it = 0
    while np.abs(obj - prv_obj) > tol:
        np.set_printoptions(suppress=True)
        grad = gradient(AgB, x_stacked, pgu_stacked)
        prv_AgB = np.copy(AgB)
        prv_obj = obj
        obj = objective(AgB, x_stacked, pgu_stacked)

        step = 0.001
        gen_grad = generalized_gradient(AgB, grad, step)
        while obj > prv_obj - step*(grad*gen_grad).sum() + step/2*np.square(gen_grad).sum() or obj > prv_obj:
            update = prv_AgB - step*grad
            step /= 2
            gen_grad = generalized_gradient(AgB, grad, step)

            # threshold A
            A = update[:,:yDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = update[:,yDim:(yDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = update[:,(yDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB[:,:yDim] = A
            AgB[:,yDim:(yDim+1)] = g
            AgB[:,(yDim+1):] = B

            obj = objective(AgB, x_stacked, pgu_stacked)

        if obj > prv_obj:
            print("Warning: increasing objective", file=sys.stderr)
            print("\tWas:", prv_obj, "Is:", obj, file=sys.stderr)
            break

        obj = objective(AgB, x_stacked, pgu_stacked)
        it += 1

        if verbose:# and it % 100 == 0:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(max_iter), file=sys.stderr)
            break
    #print("\t", it, obj)
    A = AgB[:,:yDim]
    g = AgB[:,yDim:(yDim+1)].flatten()
    B = AgB[:,(yDim+1):]

    return A, g, B


def least_squares_observed(Y, U, T, r_A=0, r_g=0, r_B=0):
    """Computes estimates of A, g, and B using least squares to solve
    the Composition LV differential equation. 

    Parameters
    ----------
        Y   : a list of T x xDim numpy arrays (multinomial observations)
        U   : a list of T x uDim numpy arrays
        T   : a list of T x 1 numpy arrays with the time of each observation
        r_A : regularization parameter for the matrix A
        r_g : regularization parameter for growth rates
        r_B : regularization parameter for the matrix B

    Returns
    -------
        A, g, B
    """
    X = estimate_latent_from_observations(Y)
    return least_squares_latent(X, U, T, r_A, r_g, r_B)



def least_squares_latent(X, U, T, r_A=0, r_g=0, r_B=0):
    """Computes estimates of A, g, and B using least squares to solve
    the Composition LV differential equation. 

    Parameters
    ----------
        X : a list of T x yDim numpy arrays (latent state estimates)
        U : a list of T x uDim numpy arrays
        T   : a list of T x 1 numpy arrays with the time of each observation
        r_A : regularization parameter for the matrix A
        r_g : regularization parameter for growth rates
        r_B : regularization parameter for the matrix B

    Returns
    -------
    """
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB_term1 = np.zeros(( xDim, yDim + 1 + uDim ))
    AgB_term2 = np.zeros(( yDim + 1 + uDim, yDim + 1 + uDim))

    for idx, (xi, ui) in enumerate(zip(X, U)):
        for t in range(1, xi.shape[0]):
            Zt = np.concatenate((xi[t], np.array([0])))
            pt  = np.exp(Zt - logsumexp(Zt))

            Zt0 = np.concatenate((xi[t-1], np.array([0])))
            pt0  = np.exp(Zt0 - logsumexp(Zt0))
            
            xt  = xi[t]
            xt0 = xi[t-1]
            
            #pt0 = pt0[:-1]
            gt0 = np.ones(1)
            ut0 = ui[t-1]
            
            pgu = np.concatenate((pt0, gt0, ut0))
            
            delT = T[idx][t] - T[idx][t-1]
            AgB_term1 += np.outer( (xt - xt0) / delT, pgu)
            AgB_term2 += np.outer(pgu, pgu)
    
    reg = np.array([r_A for i in range(yDim)] + [r_g] + [r_B for i in range(uDim)])
    reg = np.diag(reg)
    AgB = AgB_term1.dot(np.linalg.pinv(AgB_term2 + reg))
    A = AgB[:,:yDim]
    g = AgB[:,yDim:(yDim+1)].flatten()
    B = AgB[:,(yDim+1):]

    return A, g, B


def elastic_net_lotka_volterra(X, U, T, Q_inv, r_A, r_g, r_B, alpha=1, tol=1e-3, verbose=False, max_iter=100000):

    def gradient(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:,:xDim]
        g = AgB[:,xDim:(xDim+1)]
        B = AgB[:,(xDim+1):]
        grad[:,:xDim] += -2*(1-r_A)*alpha*A
        grad[:,xDim:(xDim+1)] += -2*(1-r_g)*alpha*g
        grad[:,(xDim+1):] += -2*(1-r_B)*alpha*B
        return -grad


    def generalized_gradient(AgB, grad, step):
            proximal = prv_AgB - step*grad

            # threshold A
            A = proximal[:,:xDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = proximal[:,xDim:(xDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = proximal[:,(xDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB_proximal = np.zeros(AgB.shape)
            AgB_proximal[:,:xDim] = A
            AgB_proximal[:,xDim:(xDim+1)] = g
            AgB_proximal[:,(xDim+1):] = B

            return (AgB - AgB_proximal)/step


    def objective(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*(f.dot(Q_inv)*f).sum()

        return -obj


    def stack_observations(X, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by xDim + 1 + uDim
        pgu_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                pt0 = np.exp(x[t-1])
                gt0 = np.ones(1)
                ut0 = u[t-1]

                pgu = np.concatenate((pt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t-1]
                    pgu_stacked = dt*pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))
        return x_stacked, pgu_stacked


    # assert 0 < r_A and r_A < 1
    # assert 0 < r_g and r_g < 1
    # assert 0 < r_B and r_B < 1
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, xDim + 1 + uDim ))
    x_stacked, pgu_stacked = stack_observations(X, U, T)
    prv_obj = np.inf
    obj = objective(AgB, x_stacked, pgu_stacked)

    it = 0
    while np.abs(obj - prv_obj) > tol:
        np.set_printoptions(suppress=True)
        grad = gradient(AgB, x_stacked, pgu_stacked)
        prv_AgB = np.copy(AgB)
        prv_obj = obj
        obj = objective(AgB, x_stacked, pgu_stacked)

        step = 0.001
        gen_grad = generalized_gradient(AgB, grad, step)
        while obj > prv_obj - step*(grad*gen_grad).sum() + step/2*np.square(gen_grad).sum() or obj > prv_obj:
            update = prv_AgB - step*grad
            step /= 2
            gen_grad = generalized_gradient(AgB, grad, step)

            # threshold A
            A = update[:,:xDim]
            A[A < -step*alpha*r_A] += step*alpha*r_A
            A[A > step*alpha*r_A] -= step*alpha*r_A
            A[np.logical_and(A >= -step*alpha*r_A, A <= step*alpha*r_A)] = 0

            # threshold g
            g = update[:,xDim:(xDim+1)]
            g[g < -step*alpha*r_g] += step*alpha*r_g
            g[g > step*alpha*r_g] -= step*alpha*r_g
            g[np.logical_and(g >= -step*alpha*r_g, g <= step*alpha*r_g)] = 0

            # threshold B
            B = update[:,(xDim+1):]
            B[B < -step*alpha*r_B] += step*alpha*r_B
            B[B > step*alpha*r_B] -= step*alpha*r_B
            B[np.logical_and(B >= -step*alpha*r_B, B <= step*alpha*r_B)] = 0

            AgB[:,:xDim] = A
            AgB[:,xDim:(xDim+1)] = g
            AgB[:,(xDim+1):] = B

            obj = objective(AgB, x_stacked, pgu_stacked)

        if obj > prv_obj:
            print("Warning: increasing objective", file=sys.stderr)
            print("\tWas:", prv_obj, "Is:", obj, file=sys.stderr)
            break

        obj = objective(AgB, x_stacked, pgu_stacked)
        it += 1

        if verbose:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(max_iter), file=sys.stderr)
            break

    A = AgB[:,:xDim]
    g = AgB[:,xDim:(xDim+1)].flatten()
    B = AgB[:,(xDim+1):]

    return A, g, B


def elastic_net_linear(X, U, T, Q_inv, r_A, r_g, r_B, alpha=1, tol=1e-3, verbose=False, max_iter=100000):

    def gradient(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot(f.T.dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:,:xDim]
        g = AgB[:,xDim:(xDim+1)]
        B = AgB[:,(xDim+1):]
        grad[:,:xDim] += -2*alpha*(1-r_A)*A
        grad[:,xDim:(xDim+1)] += -2*alpha*(1-r_g)*g
        grad[:,(xDim+1):] += -2*alpha*(1-r_B)*B
        return -grad


    def generalized_gradient(AgB, grad, step):
            proximal = prv_AgB - step*grad

            # threshold A
            A = proximal[:,:xDim]
            A[A < -step*r_A] += step*r_A
            A[A > step*r_A] -= step*r_A
            A[np.logical_and(A >= -step*r_A, A <= step*r_A)] = 0

            # threshold g
            g = proximal[:,xDim:(xDim+1)]
            g[g < -step*r_g] += step*r_g
            g[g > step*r_g] -= step*r_g
            g[np.logical_and(g >= -step*r_g, g <= step*r_g)] = 0

            # threshold B
            B = proximal[:,(xDim+1):]
            B[B < -step*r_B] += step*r_B
            B[B > step*r_B] -= step*r_B
            B[np.logical_and(B >= -step*r_B, B <= step*r_B)] = 0

            AgB_proximal = np.zeros(AgB.shape)
            AgB_proximal[:,:xDim] = A
            AgB_proximal[:,xDim:(xDim+1)] = g
            AgB_proximal[:,(xDim+1):] = B

            return (AgB - AgB_proximal)/step


    def objective(AgB, x_stacked, pgu_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*(f.dot(Q_inv)*f).sum()

        return -obj

    def stack_observations(X, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by xDim + 1 + uDim
        pgu_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                xt0 = x[t-1]
                gt0 = np.ones(1)
                ut0 = u[t-1]
                pgu = np.concatenate((xt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = x[t] - x[t-1]
                    pgu_stacked = dt*pgu

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))

        return x_stacked, pgu_stacked


    # assert 0 < r_A and r_A < 1
    # assert 0 < r_g and r_g < 1
    # assert 0 < r_B and r_B < 1
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, xDim + 1 + uDim ))
    x_stacked, pgu_stacked = stack_observations(X, U, T)
    prv_obj = np.inf
    obj = objective(AgB, x_stacked, pgu_stacked)

    it = 0
    while np.abs(obj - prv_obj) > tol:
        np.set_printoptions(suppress=True)
        grad = gradient(AgB, x_stacked, pgu_stacked)
        prv_AgB = np.copy(AgB)
        prv_obj = obj
        obj = objective(AgB, x_stacked, pgu_stacked)

        step = 0.001
        gen_grad = generalized_gradient(AgB, grad, step)
        while obj > prv_obj - step*(grad*gen_grad).sum() + step/2*np.square(gen_grad).sum() or obj > prv_obj:
            update = prv_AgB - step*grad
            step /= 2

            # threshold A
            A = update[:,:xDim]
            A[A < -step*r_A] += step*r_A
            A[A > step*r_A] -= step*r_A
            A[np.logical_and(A >= -step*r_A, A <= step*r_A)] = 0

            # threshold g
            g = update[:,xDim:(xDim+1)]
            g[g < -step*r_g] += step*r_g
            g[g > step*r_g] -= step*r_g
            g[np.logical_and(g >= -step*r_g, g <= step*r_g)] = 0

            # threshold B
            B = update[:,(xDim+1):]
            B[B < -step*r_B] += step*r_B
            B[B > step*r_B] -= step*r_B
            B[np.logical_and(B >= -step*r_B, B <= step*r_B)] = 0

            AgB[:,:xDim] = A
            AgB[:,xDim:(xDim+1)] = g
            AgB[:,(xDim+1):] = B

            obj = objective(AgB, x_stacked, pgu_stacked)

        if obj > prv_obj:
            print("Warning: increasing objective", file=sys.stderr)
            print("\tWas:", prv_obj, "Is:", obj, file=sys.stderr)
            break

        obj = objective(AgB, x_stacked, pgu_stacked)
        it += 1

        if verbose and it % 100 == 0:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(max_iter), file=sys.stderr)
            break
    #print("\t", it, obj)
    A = AgB[:,:xDim]
    g = AgB[:,xDim:(xDim+1)].flatten()
    B = AgB[:,(xDim+1):]

    return A, g, B