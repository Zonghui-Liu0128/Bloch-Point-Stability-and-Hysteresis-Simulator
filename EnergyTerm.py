import abc
import numpy as np

# Some important hyperparameters
mu0 = 4 * np.pi * 1e-7
Ms = 3.84e5
nx, ny, nz = (10, 10, 10)
dx, dy, dz = (5e-9, 5e-9, 5e-9)
A = 8.78e-12
D = 1.58e-3
B = 0.1


class EnergyTerm(abc.ABC):
    @abc.abstractmethod
    def effective_field(self, m):
        pass

    def energy_density(self, m):
        tmp = -0.5 * mu0 * Ms * np.multiply(m.field(), self.effective_field(m))
        return np.sum(tmp, axis=3, keepdims=True)

    def energy(self, m):
        dV = dx * dy * dz
        return np.sum(self.energy_density(m) * dV)


class Exchange(EnergyTerm):
    def __init__(self, A):
        self.A = A

    def effective_field(self, m):
        return ((2 * self.A) / (mu0 * Ms)) * m.laplace()


class Zeeman(EnergyTerm):
    def __init__(self, H):
        self.H = H

    def effective_field(self, m):
        return np.tile(self.H, nx * ny * nz).reshape((nx, ny, nz, 3))

    def energy_density(self, m):
        tmp = -mu0 * Ms * np.multiply(m.field(), self.effective_field(m))
        return np.sum(tmp, axis=3, keepdims=True)


class DMI(EnergyTerm):
    def __init__(self, D):
        self.D = D

    def effective_field(self, m):
        return -((2 * self.D) / (mu0 * Ms)) * m.curl_3D()


class M:
    def __init__(self, m):
        self.m = m

    def field(self):
        return self.m

    def set_field(self, m_new):
        self.m = m_new

    # This function is discretized on the boundary using a first order difference
    def laplace(self):
        m_laplace = np.zeros_like(self.m)
        m_pad = np.pad(self.m, ((1, 1), (1, 1), (1, 1), (0, 0)), 'edge')

        m_laplace += (m_pad[0:-2, 1:-1, 1:-1] + m_pad[2:, 1:-1, 1:-1] - 2 * m_pad[1:-1, 1:-1, 1:-1]) / dx ** 2
        m_laplace += (m_pad[1:-1, 0:-2, 1:-1] + m_pad[1:-1, 2:, 1:-1] - 2 * m_pad[1:-1, 1:-1, 1:-1]) / dy ** 2
        m_laplace += (m_pad[1:-1, 1:-1, 0:-2] + m_pad[1:-1, 1:-1, 2:] - 2 * m_pad[1:-1, 1:-1, 1:-1]) / dz ** 2

        return m_laplace

    def curl_1D(self):
        curl_1D = np.zeros_like(self.m)

        dMz_dy = np.zeros((nx, ny, nz))
        dMx_dy = np.zeros((nx, ny, nz))

        bc_left = (D / (2 * A)) * np.cross(np.cross((0, -1, 0), self.m[:, 0, :]), (0, 0, 1))
        m_gl = - dy * bc_left

        bc_right = (D / (2 * A)) * np.cross(np.cross((0, 1, 0), self.m[:, -1, :]), (0, 0, 1))
        m_gr = dy * bc_right

        m_pad = np.pad(self.m, ((0, 0), (1, 1), (0, 0), (0, 0)), 'empty')
        m_pad[:, 0, :] = m_gl
        m_pad[:, -1, :] = m_gr

        dMz_dy[:, :, :] += (m_pad[:, 2:, :, 2] - m_pad[:, :-2, :, 2]) / (2 * dy)
        dMx_dy[:, :, :] += (m_pad[:, 2:, :, 0] - m_pad[:, :-2, :, 0]) / (2 * dy)

        curl_1D[:, :, :, 0] += dMz_dy
        curl_1D[:, :, :, 2] += -dMx_dy
        return curl_1D

    def curl_3D(self):
        curl = np.zeros_like(self.m)

        dzdy = np.zeros((nx, ny, nz))
        dydz = np.zeros((nx, ny, nz))
        dxdz = np.zeros((nx, ny, nz))
        dzdx = np.zeros((nx, ny, nz))
        dxdy = np.zeros((nx, ny, nz))
        dydx = np.zeros((nx, ny, nz))

        bc_left = (D / (2 * A)) * np.cross(np.cross((0, -1, 0), self.m[:, 0, :]), (0, 0, 1))
        m_gl = - dy * bc_left

        bc_right = (D / (2 * A)) * np.cross(np.cross((0, 1, 0), self.m[:, -1, :]), (0, 0, 1))
        m_gr = + dy * bc_right

        bc_front = (D / (2 * A)) * np.cross(np.cross((1, 0, 0), self.m[-1, :, :]), (0, 1, 0))
        m_gf = + dx * bc_front

        bc_back = (D / (2 * A)) * np.cross(np.cross((-1, 0, 0), self.m[0, :, :]), (0, 1, 0))
        m_gb = - dx * bc_back

        bc_up = (D / (2 * A)) * np.cross(np.cross((0, 0, 1), self.m[:, :, -1]), (1, 0, 0))
        m_gu = + dz * bc_up

        bc_down = (D / (2 * A)) * np.cross(np.cross((0, 0, -1), self.m[:, :, 0]), (1, 0, 0))
        m_gd = - dz * bc_down

        m_pad = np.pad(self.m, ((1, 1), (1, 1), (1, 1), (0, 0)), 'empty')
        m_pad[1:-1, 0, 1:-1] = m_gl
        m_pad[1:-1, -1, 1:-1] = m_gr
        m_pad[0, 1:-1, 1:-1] = m_gb
        m_pad[-1, 1:-1, 1:-1] = m_gf
        m_pad[1:-1, 1:-1, 0] = m_gd
        m_pad[1:-1, 1:-1, -1] = m_gu

        dzdy[...] += (m_pad[1:-1, 2:, 1:-1, 2] -
                      m_pad[1:-1, :-2, 1:-1, 2]) / (2 * dy)

        dydz[...] += (m_pad[1:-1, 1:-1, 2:, 1] -
                      m_pad[1:-1, 1:-1, :-2, 1]) / (2 * dz)

        dxdz[...] += (m_pad[1:-1, 1:-1, 2:, 0] -
                      m_pad[1:-1, 1:-1, :-2, 0]) / (2 * dz)
        dzdx[...] += (m_pad[2:, 1:-1, 1:-1, 2] -
                      m_pad[:-2, 1:-1, 1:-1, 2]) / (2 * dx)

        dxdy[...] += (m_pad[1:-1, 2:, 1:-1, 0] -
                      m_pad[1:-1, :-2, 1:-1, 0]) / (2 * dy)
        dydx[...] += (m_pad[2:, 1:-1, 1:-1, 1] -
                      m_pad[:-2, 1:-1, 1:-1, 1]) / (2 * dx)

        curl[..., 0] += dzdy - dydz
        curl[..., 1] += dxdz - dzdx
        curl[..., 2] += dydx - dxdy
        return curl
