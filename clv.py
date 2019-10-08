import numpy as np
import scipy.stats as stats
import sys

from scipy.special import gammaln
from scipy.special import logsumexp

from blk_tridiag_inv import compute_blk_tridiag
from blk_tridiag_inv import compute_blk_tridiag_inv_b

from optimizers import least_squares_latent, least_squares_observed


def elastic_net(X, U, T, Q_inv, r_A, r_g, r_B, tol=1e-3, verbose=False, max_iter=100000):

    def gradient(AgB, x_stacked, pgu_stacked, dt_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        grad = Q_inv.dot((dt_stacked*f.T).dot(pgu_stacked))

        # l2 regularization terms
        A = AgB[:,:xDim]
        g = AgB[:,xDim:(xDim+1)]
        B = AgB[:,(xDim+1):]
        grad[:,:xDim] += -2*(1-r_A)*A
        grad[:,xDim:(xDim+1)] += -2*(1-r_g)*g
        grad[:,(xDim+1):] += -2*(1-r_B)*B
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


    def objective(AgB, x_stacked, pgu_stacked, dt_stacked):
        f = x_stacked - AgB.dot(pgu_stacked.T).T
        obj = -0.5*((dt_stacked*f.T).T.dot(Q_inv)*f).sum()

        return -obj


    def stack_observations(X, U, T):
        # number of observations by xDim
        x_stacked = None
        # number of observations by xDim + 1 + uDim
        pgu_stacked = None
        # number of observations - 1
        dt_stacked = None
        for x, u, times in zip(X, U, T):
            for t in range(1, times.size):
                dt = times[t] - times[t-1]
                zt0 = np.concatenate((x[t-1], np.array([0])))
                pt0 = np.exp(zt0 - logsumexp(zt0))[:-1]
                gt0 = np.ones(1)
                ut0 = u[t-1]
                pgu = np.concatenate((pt0, gt0, ut0))

                if x_stacked is None:
                    x_stacked = (x[t] - x[t-1])
                    pgu_stacked = dt*pgu
                    dt_stacked = np.array([dt])

                else:
                    x_stacked = np.vstack((x_stacked, x[t] - x[t-1]))
                    pgu_stacked = np.vstack((pgu_stacked, dt*pgu))
                    dt_stacked = np.concatenate((dt_stacked, np.array([dt])))

        return x_stacked, pgu_stacked, dt_stacked


    assert 0 < r_A and r_A < 1
    assert 0 < r_g and r_g < 1
    assert 0 < r_B and r_B < 1
    xDim = X[0].shape[1]
    yDim = xDim + 1
    uDim = U[0].shape[1]
    AgB = np.zeros(( xDim, xDim + 1 + uDim ))
    x_stacked, pgu_stacked, dt_stacked = stack_observations(X, U, T)
    prv_obj = np.inf
    obj = objective(AgB, x_stacked, pgu_stacked, dt_stacked)

    it = 0
    while np.abs(obj - prv_obj) > tol:
        np.set_printoptions(suppress=True)
        grad = gradient(AgB, x_stacked, pgu_stacked, dt_stacked)
        prv_AgB = np.copy(AgB)
        prv_obj = obj
        obj = objective(AgB, x_stacked, pgu_stacked, dt_stacked)

        step = 0.001
        gen_grad = generalized_gradient(AgB, grad, step)
        while obj > prv_obj - step*(grad*gen_grad).sum() + step/2*np.square(gen_grad).sum():
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

            obj = objective(AgB, x_stacked, pgu_stacked, dt_stacked)

        if obj > prv_obj:
            print("Warning: increasing objective", file=sys.stderr)
            print("\tWas:", prv_obj, "Is:", obj, file=sys.stderr)
            break

        obj = objective(AgB, x_stacked, pgu_stacked, dt_stacked)
        it += 1

        if verbose and it % 5000 == 0:
            print("\t", it, obj)

        if it > max_iter:
            print("Warning: maximum number of iterations ({}) reached".format(max_iter), file=sys.stderr)
            break
    #print("\t", it, obj)
    A = AgB[:,:xDim]
    g = AgB[:,xDim:(xDim+1)].flatten()
    B = AgB[:,(xDim+1):]

    return A, g, B


class CompositionalLV(object):

    def __init__(self, obs_dim, effect_dim, optimize_Q=True):
        latent_dim = obs_dim - 1
        self.latent_dim = latent_dim
        self.obs_dim    = obs_dim
        self.effect_dim = effect_dim
        self.optimize_Q = optimize_Q

        ########################################
        ###         Model Parameters         ###
        ########################################
        # state space matrix / transition matrix
        self.A  = np.random.normal(size=(latent_dim, latent_dim)) #np.zeros( (latent_dim, latent_dim) ) #
        # external effects matrix
        self.B  = np.random.normal(size=(latent_dim, effect_dim))#np.zeros( (latent_dim, effect_dim) )
        # latent growth rates
        self.g  = np.random.normal(size=latent_dim)# np.zeros( latent_dim ) #np.random.uniform(size=(latent_dim))
        # initial mean
        self.mu = np.zeros(latent_dim)
        # state space covariance
        self.Q  = 0.1*np.eye(latent_dim)
        # initial covariance
        self.Q0 = 0.1*np.eye(latent_dim)


    def multinomial_log_pmf(self, y, p):
        """Compute the log pmf of the multinomial distribution.

        Parameters
        ----------
            x : obs_dim numpy array of observed counts
            p : vector of probabilities for each observation
        """
        # p = p + 1e-8
        # p = p / p.sum()
        return gammaln(y.sum() + 1) - gammaln(y+1).sum() + (y*np.log(p)).sum()


    def simulate(self, T, u, N):
            """Simulate a sequence of length T.

            Parameters
            ----------
                T  : number of time points to simulate
                u  : T x effect_dim numpy array of external effects
                N  : Poisson parameter for number of observed reads

            Returns
            -------
                x  : T x latent_dim numpy array of latent states
                y  : T x obs_dim numpy array of multinomial observations

            """
            x = []
            y = []
            mu  = self.mu
            cov = self.Q0
            for t in range(T):
                xt = np.random.multivariate_normal(mean=mu, cov=cov)

                # increase dimension by 1
                xt1 = np.concatenate((xt, np.array([0])))
                pt = np.exp(xt1 - logsumexp(xt1))

                # simulate total number of reads with over-dispersion
                logN = np.random.normal(loc=np.log(N), scale=0.5)
                Nt = np.random.poisson(np.exp(logN))
                yt = np.random.multinomial(Nt, pt).astype(float)

                x.append(xt)
                y.append(yt)

                mu  = xt + self.g + self.A.dot(pt[:-1]) + self.B.dot(u[t])
                cov = self.Q

            x = np.array(x)
            y = np.array(y)
            return x, y


    def _compute_gradient_block(self, x, y, u, times):
        t_pts = times.shape[0]
        lat_dim = self.latent_dim
        eff_dim = self.effect_dim
        A = self.A
        B = self.B
        Q_inv = np.linalg.pinv(self.Q)
        Q0_inv = np.linalg.pinv(self.Q0)
        g = self.g
        grad = np.zeros((times.size, self.latent_dim))

        for t in range(t_pts):
            if t == 0:
                y0 = np.copy(y[0])
                y0[y0 == 0] = 1
                mu = np.log(y0[:-1]/y0[-1])
                dt1 = times[t+1] - times[t]
                pt = (y[t] / y[t].sum())[:-1]
                gt = -Q0_inv.dot(x[t] - mu) + dt1*Q_inv.dot(x[t+1] - x[t] - dt1*(A.dot(pt) + B.dot(u[t]) + g)) + \
                      y[t][:-1] - y[t].sum()*np.exp(x[t]) / (1 + np.exp(x[t]).sum())
                grad[t] = gt
            elif t > 0 and t < times.shape[0]-1:
                pt0 = (y[t-1] / y[t-1].sum())[:-1]
                pt = (y[t] / y[t].sum())[:-1]
                dt = times[t] - times[t-1]
                dt1 = times[t+1] - times[t]
                gt = -dt*Q_inv.dot(x[t] - x[t-1] - dt*(A.dot(pt0) + B.dot(u[t-1]) + g)) + \
                    dt1*Q_inv.dot(x[t+1] - x[t] - dt1*(A.dot(pt) + B.dot(u[t]) + g)) + \
                    y[t][:-1] - y[t].sum()*np.exp(x[t]) / (1 + np.exp(x[t]).sum())
                grad[t] = gt
            else:
                pt0 = (y[t-1] / y[t-1].sum())[:-1]
                dt = times[t] - times[t-1]
                gt = -dt*Q_inv.dot(x[t] - x[t-1] - dt*(A.dot(pt0) + B.dot(u[t-1]) + g)) + \
                      y[t][:-1] - y[t].sum()*np.exp(x[t]) / (1 + np.exp(x[t]).sum())
                grad[t] = gt

        return grad


    def _compute_hessian_block_diag(self, x, y, u, times):
        t_pts = times.shape[0]
        A = self.A
        B = self.B
        Q_inv = np.linalg.pinv(self.Q)
        Q0_inv = np.linalg.pinv(self.Q0)
        g = self.g
        lat_dim = self.latent_dim
        eff_dim = self.effect_dim

        # Diagonal entries
        AA = np.zeros((t_pts, lat_dim, lat_dim))
        # Upper diagonal entries
        BB = np.zeros((t_pts-1, lat_dim, lat_dim))

        for t in range(t_pts-1):
            dt = 1 if t == 0 else times[t] - times[t-1]
            dt1 = times[t+1] - times[t]

            # Upper Diagonal
            BB[t] = dt1*Q_inv

            n = y[t].sum()
            H = -n*(1/(1 + np.exp(x[t]).sum()) - 1/np.square((1 + np.exp(x[t]).sum()))*np.outer(np.exp(x[t]), np.exp(x[t])))
            
            # Diagonal
            if t == 0:
                AA[t] = -Q0_inv - dt1*Q_inv + H
            else:
                AA[t] = -dt*Q_inv - dt1*Q_inv + H


        # Last Time Point
        dt = times[t_pts-1] - times[t_pts-2]
        # hessian of observations
        n = y[t].sum()
        t = t_pts-1
        H = -n*(1/(1 + np.exp(x[t]).sum()) - 1/np.square((1 + np.exp(x[t]).sum()))*np.outer(np.exp(x[t]), np.exp(x[t])))
 
        AA[t_pts-1] = -dt*Q_inv + H
        return AA, BB


    def _compute_obj(self, x, y, u, times, r_A=0, r_g=0, r_B=0):
        norm = stats.multivariate_normal
        t_pts = times.shape[0]

        lat_dim = self.latent_dim
        eff_dim = self.effect_dim
        A = self.A
        B = self.B
        Q = self.Q
        Q0 = self.Q0
        Q0_inv = np.linalg.pinv(self.Q0)
        Q_inv = np.linalg.pinv(self.Q)
        g = self.g
        y0 = np.copy(y[0])
        y0[y0 == 0] = 1
        mu = np.log(y0[:-1]/y0[-1])

        # obj = 0
        # for t in range(t_pts):
        #     if t == 0:
        #         zt = np.concatenate((x[t], np.array([0])))
        #         pt = np.exp(zt - logsumexp(zt))
        #         obj += norm.logpdf(x[t], mu, Q0) 
        #         obj += self.multinomial_log_pmf(y[t], pt)
        #     else:
        #         dt = times[t] - times[t-1]
        #         zt0 = np.concatenate((x[t-1], np.array([0])))
        #         pt0 = y[t-1]/y[t-1].sum()
        #         ut0 = u[t-1]

        #         zt = np.concatenate((x[t], np.array([0])))
        #         pt = np.exp(zt - logsumexp(zt))

        #         f = np.copy(x[t] - x[t-1] - dt*(A.dot(pt0[:-1]) + B.dot(ut0) + g))
        #         obj += -0.5*f.T.dot(dt*Q_inv).dot(f)
        #         obj += norm.logpdf(np.zeros(lat_dim), np.zeros(lat_dim), dt*Q)

        #         # obj += norm.logpdf(x[t], x[t-1] + dt*(np.dot(A, pt0[:-1]) + np.dot(B, ut0) + g), dt*Q)# + \
        #         obj += self.multinomial_log_pmf(y[t], pt)

        obj = 0
        z = np.hstack((x, np.zeros((x.shape[0], 1))))
        p = np.exp(z.T - logsumexp(z, axis=1)).T

        # we're below the precision of the machine for low
        # probability taxa and should stop optimizing
        if np.any(p == 0):
            return np.nan

        p0 = (y / y.sum(axis=1, keepdims=True))[:-1,:-1]
        dt = times[1:] - times[:-1]

        obj += -0.5*(x[0] - mu).dot(Q0_inv.dot(x[0] - mu))
        f = (x[1:].T - x[:-1].T - dt*(A.dot(p0.T) + B.dot(u[:-1].T) + g.reshape((g.size, 1))))
        obj += -0.5*((dt*f).T.dot(Q_inv) * f.T).sum()
        obj += gammaln(y.sum(axis=1) + 1).sum(axis=0) - gammaln(y+1).sum() + (y*np.log(p)).sum()

        # from covariance
        obj += -lat_dim*0.5*np.log(2*np.pi) - 0.5*np.linalg.slogdet(self.Q0)[1]

        obj += -0.5*lat_dim*(times.size-1)*np.log(2*np.pi) \
               -0.5*lat_dim*np.log(dt).sum() - 0.5*(times.size-1)*np.log(np.linalg.det(Q))
        
        # l1 regularization terms
        obj += -r_A*np.abs(A).sum() - r_g*np.abs(g).sum() - r_B*np.abs(B).sum()
        # l2 regularization terms
        obj += -(1-r_A)*np.square(A).sum() - (1-r_g)*np.square(g).sum() - (1-r_B)*np.square(B).sum()

        return obj


    def newton_raphson(self, y, u, times, x):
        """
        """
        tol = 1E-2
        prv = -1
        nxt = 0.
        x = np.copy(x)

        """
        Newton Step is zt = z_prv - H_inv g
            => H_inv g = z_prv - zt
            => We need to solve:
               H(z_prv - zt) = g
            Let y = z_prv - zt
            Solve:
               Hy = g
            Set:
               zt = z_prv - y
        """
        it = 0
        max_iter = 1
        while nxt - prv > tol and not np.isnan(nxt):
            prv = self._compute_obj(x, y, u, times)

            it += 1
            g = self._compute_gradient_block(x, y, u, times)
            step_size = 1.
            H_AA, H_BB = self._compute_hessian_block_diag(x, y, u, times)
            D_inv, OD_inv, S = compute_blk_tridiag(H_AA, H_BB)
            s = compute_blk_tridiag_inv_b(S, D_inv, g)
            x_prv = np.copy(x)
            x = x_prv - s
            x = np.clip(x, -20, 20)
            nxt = self._compute_obj(x, y, u, times)
            while nxt < prv:
                step_size /= 10.
                s = compute_blk_tridiag_inv_b(S, D_inv, step_size*g)
                x = x_prv - s
                x = np.clip(x, -20, 20)
                nxt = self._compute_obj(x, y, u, times)

            if it >= max_iter:
                break
            #print("\t", nxt)

        H_AA, H_BB = self._compute_hessian_block_diag(x, y, u, times)
        D_inv, OD_inv, S = compute_blk_tridiag(H_AA, H_BB)

        cov_D  = -D_inv # diagonal block of covariance matrix
        cov_OD = -OD_inv
        return x, cov_D, cov_OD


    def update_state_space_estimates(self, Y, U, T, X):
        """Compute the E step of the EM algorithm.
        """
        lat_dim = self.latent_dim
        smoothed_means = []
        for y, u, times, x in zip(Y, U, T, X):
            xt_T, cov_D, cov_OD = self.newton_raphson(y, u, times, np.copy(x))
            smoothed_means.append(xt_T)

        return smoothed_means


    def update_parameter_estimates(self, smoothed_means, U, T, r_A, r_g, r_B):
        self.A, self.g, self.B = elastic_net(smoothed_means, U, T, np.linalg.pinv(self.Q), r_A, r_g, r_B, tol=1e-2, verbose=False)

        if self.optimize_Q:
            self.Q = self.update_Q(smoothed_means, U, T)


    def update_Q(self, X, U, T):
        A = self.A
        B = self.B
        g = self.g
        Q = np.zeros(self.Q.shape)

        n = 0
        for idx, x in enumerate(X):
            for t in range(1, x.shape[0]):
                dt = T[idx][t] - T[idx][t-1]
                zt0 = np.concatenate((x[t-1], np.array([0])))
                pt0 = np.exp(zt0 - logsumexp(zt0))
                ut0 = U[idx][t-1]

                tmp = (x[t] - x[t-1] - dt*(A.dot(pt0[:-1]) + B.dot(ut0) + g))
                Q += np.outer(tmp, tmp)/dt
                n += 1
        Q /= n
        Q = 0.5*Q + 0.5*Q.T

        return np.diag(Q)*np.eye(self.latent_dim)


    def train(self, Y, U, T, r_A, r_g, r_B, max_iter=1000):
        prv_obj = 0
        obj = 10
        it = 0

        smoothed_means = []
        for y in Y:
            x = np.zeros((y.shape[0], y.shape[1]-1))
            for t,yt in enumerate(y):
                yt = np.copy(yt)
                yt[yt == 0] = 1
                yt /= yt.sum()
                x[t] = np.log(yt[:-1] / yt[-1])
            smoothed_means.append(x)

        while np.abs(obj - prv_obj) > 1e-2:
            #print("0", self.compute_objective(smoothed_means, Y, U, T, r_A, r_g, r_B))
            smoothed_means = self.update_state_space_estimates(Y, U, T, smoothed_means)
            # stop if the estimated state space does not change
            ss_obj = self.compute_objective(smoothed_means, Y, U, T, r_A, r_g, r_B)
            if np.abs(ss_obj - obj) < 1e-4:
                break
            #print("1", ss_obj)

            self.update_parameter_estimates(smoothed_means, U, T, r_A, r_g, r_B)
            prv_obj = obj
            obj = self.compute_objective(smoothed_means, Y, U, T, r_A, r_g, r_B)
            #print("2", obj)
            it+=1

            print("it:", it, "obj:", obj)

            if it > max_iter:
                break
        return self.A, self.g, self.B


    def compute_objective(self, X, Y, U, T, r_A, r_g, r_B):
        obj = 0
        for x, y, u, t in zip(X, Y, U, T):
            obj += self._compute_obj(x, y, u, t, r_A, r_g, r_B)
        return obj
