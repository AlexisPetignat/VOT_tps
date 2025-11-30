import numpy as np


class KalmanFilter:
    def __init__(
        self,
        dt: float,
        u_x: float,
        u_y: float,
        std_acc: float,
        x_std_meas: float,
        y_std_meas: float,
    ):
        # Compute A
        A = np.identity(4)
        A[0, 2] = dt
        A[1, 3] = dt

        # Compute B
        tmp = (dt**2) / 2
        B = np.array([[tmp, 0], [0, tmp], [dt, 0], [0, dt]])

        # Compute H
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Compute Q
        dt2 = dt**2
        dt3 = (dt**3) / 2
        dt4 = (dt**4) / 4
        Q = np.array(
            [[dt4, 0, dt3, 0], [0, dt4, 0, dt3], [dt3, 0, dt2, 0], [0, dt3, 0, dt2]]
        )

        # Compute R
        R = np.identity(2) * (x_std_meas**2)
        R[1, 1] = y_std_meas**2

        # Assign
        self.u = (u_x, u_y)
        self.x_k: np.ndarray = np.zeros(shape=(4))
        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.H: np.ndarray = H
        self.Q: np.ndarray = Q * std_acc
        self.R: np.ndarray = R
        self.P: np.ndarray = np.identity(4)

    def predict(self):
        self.x_k = self.A @ self.x_k + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_k[0], self.x_k[1]

    def update(self, z_k: np.ndarray):
        P_k = self.P
        X_k = self.x_k

        # Kalman gain
        S_k = self.H @ P_k @ self.H.T + self.R
        K_k = P_k @ self.H.T @ np.linalg.inv(S_k)

        # update
        self.x_k = X_k + (K_k @ (z_k.T - self.H @ X_k)[0])
        self.P = (np.identity(K_k.shape[0]) - (K_k @ self.H)) @ P_k
        return self.x_k[0], self.x_k[1]
