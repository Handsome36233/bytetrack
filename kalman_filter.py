import numpy as np

def cho_solve(A, b):
    chol_factor, lower = A
    b = np.asarray(b)
    if lower:
        y = np.zeros_like(b)
        for i in range(chol_factor.shape[0]):
            y[i] = (b[i] - np.dot(chol_factor[i, :i], y[:i])) / chol_factor[i, i]
        x = np.zeros_like(y)
        for i in range(chol_factor.shape[0] - 1, -1, -1):
            x[i] = (y[i] - np.dot(chol_factor[i + 1:, i], x[i + 1:])) / chol_factor[i, i]
    else:
        y = np.zeros_like(b)
        for i in range(chol_factor.shape[0]):
            y[i] = (b[i] - np.dot(chol_factor[:i, i], y[:i])) / chol_factor[i, i]
        x = np.zeros_like(y)
        for i in range(chol_factor.shape[0] - 1, -1, -1):
            x[i] = (y[i] - np.dot(chol_factor[i, i + 1:], x[i + 1:])) / chol_factor[i, i]
    return x

def cho_factor(A, lower=True):
    A = np.asarray(A)
    L = np.linalg.cholesky(A)
    if lower:
        return L, lower
    else:
        return L.T, lower

def solve_triangular(A, b, lower=False, overwrite_b=False):
    A = np.asarray(A)
    b = np.array(b, copy=not overwrite_b)
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "A 必须是方阵"
    assert b.shape[0] == n, "b 的行数必须与 A 匹配"
    if b.ndim == 1:
        b = b[:, np.newaxis]
    x = b if overwrite_b else np.zeros_like(b, dtype=A.dtype)
    if lower:
        for i in range(n):
            for j in range(b.shape[1]):
                x[i, j] = (b[i, j] - np.dot(A[i, :i], x[:i, j])) / A[i, i]
    else:
        for i in range(n - 1, -1, -1):
            for j in range(b.shape[1]):
                x[i, j] = (b[i, j] - np.dot(A[i, i + 1:], x[i + 1:, j])) / A[i, i]
    if x.shape[1] == 1:
        return x.ravel()
    return x

class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        # chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        # kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        chol_factor, lower = cho_factor(projected_cov, lower=True)
        kalman_gain = cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            # z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            z = solve_triangular(cholesky_factor, d.T, lower=True, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
