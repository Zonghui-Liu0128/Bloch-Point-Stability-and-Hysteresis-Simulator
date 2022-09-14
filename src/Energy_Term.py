import numpy as np
import abc

mu0 = 4 * np.pi * 1e-7
B = -0.1
D = {'bottom': -1.58e-3, 'top': 1.58e-3}
panel = 4
A = 8.78e-12

Ms = 3.84e5
nx, ny, nz = (30, 30, 6)
dx, dy, dz = (5e-9, 5e-9, 5e-9)


class EnergyTerm(abc.ABC):
    @abc.abstractmethod
    def effective_field(self, m):
        pass

    def energy_density(self, m):
        '''
        It calculate the energy density using the dot product of the
        magnetization and the effective field.

        Parameters
        ----------
        m
                the magnetisation vector

        Returns
        -------
                The energy density is being returned.

        '''
        tmp1 = np.multiply(m.get_m(), self.effective_field(m))
        tmp2 = -0.5 * mu0 * np.multiply(m.get_Ms(), tmp1)
        return np.sum(tmp2, axis=3, keepdims=True)

    def energy(self, m):
        '''
        It calculates the energy of the system by summing
        the energy density over the entire volume

        Parameters
        ----------
        m
                the magnetization

        Returns
        -------
                The energy of the system.

        '''
        dV = dx * dy * dz
        return np.sum(self.energy_density(m) * dV)


# It calculates the effective field, energy density and energy due to
# exchange interactions
class Exchange(EnergyTerm):
    def __init__(self, A):
        self.A = A

    def effective_field(self, m):
        '''
        The effective field is the Laplacian of the magnetization
        multiplied by the exchange constant divided by the
        magnetic permeability of free space

        Parameters
        ----------
        m
                the magnetisation
        Returns
        -------
                The effective field is being returned.
        '''

        if (np.allclose(m.get_M(), 0)):
            return np.zeros_like(m.get_M())
        else:
            return np.multiply((2 * self.A) / (mu0 * m.get_Ms()), m.laplace())


# It calculates the effective field, energy density and energy due to
# zeeman interactions
class Zeeman(EnergyTerm):
    def __init__(self, H):
        self.H = H

    def effective_field(self, m):
        '''
        It takes a 3D array of magnetization values and returns
        a 3D array of effective field values

        Parameters
        ----------
        m
                the magnetization

        Returns
        -------
                The effective field is being returned.

        '''
        return np.tile(self.H, nx * ny * nz).reshape((nx, ny, nz, 3))

    def energy_density(self, m):
        '''
        The energy density is the negative of the dot product
        of the magnetization and the effective field

        Parameters
        ----------
        m
                the magnetisation vector

        Returns
        -------
                The energy density is being returned.

        '''
        tmp1 = np.multiply(m.get_m(), self.effective_field(m))
        tmp = -mu0 * np.multiply(m.get_Ms(), tmp1)
        return np.sum(tmp, axis=3, keepdims=True)


# It calculates the effective field, energy density and energy due to
# Dzyaloshinskii-Moriya interactions
class DMI(EnergyTerm):
    def __init__(self, D, panel):
        '''The function takes in a dictionary of diffusion coefficients
        and a panel number, and returns a diffusion coefficient
        array with the bottom half of the array set to the bottom
        diffusion coefficient and the top half of the array set to
        the top diffusion coefficient.

        Parameters
        ----------
        D
                a dictionary of diffusion coefficients for
                the top and bottom panels
        panel
                the number of layers in the bottom panel

        '''
        self.D = np.zeros((nx, ny, nz, 1))
        self.D[:, :, 0:panel] = D['bottom']
        self.D[:, :, panel:] = D['top']

    def effective_field(self, m):
        '''The effective field is the negative of the curl
        of the magnetization, multiplied by the ratio of
        the exchange constant to the saturation magnetization

        Parameters
        ----------
        m
                the magnetisation vector

        Returns
        -------
                The effective field is being returned.

        '''
        if (np.allclose(m.get_M(), 0)):
            return np.zeros_like(m.get_M())
        else:
            return np.multiply(-((2 * self.D) / (mu0 * m.get_Ms())), m.curl())


# The class M is a class that contains the magnetization M and its
# associated functions.
class M:
    def __init__(self, M):
        '''
        Parameters
        ----------
        M
                The number of hidden states.

        '''
        self.M = M

    def get_M(self):
        '''This function returns the value of the attribute M

        Returns
        -------
                The value of the attribute M.

        '''
        return self.M

    def get_Ms(self):
        '''It takes the norm of the last axis of the input array,
        and returns an array with the same shape as the input array,
        except the last axis is replaced with the norm of t
        he last axis of the input array

        Returns
        -------
                The norm of the M matrix.

        '''
        Ms = np.expand_dims(
            np.linalg.norm(self.M, axis=3), axis=3)
        return Ms

    def get_m(self):
        '''If the magnetization is zero, return a zero magnetization,
        otherwise return the magnetization divided by the
        saturation magnetization

        Returns
        -------
                The magnetization per unit volume.

        '''
        if (np.allclose(self.get_M(), 0)):
            return np.zeros_like(self.get_M())
        else:
            return self.M / self.get_Ms()

    def set_M(self, M_new):
        '''**The function set_M() takes as input a new value for M,
        and sets the value of the M attribute of the object to this
        new value.**

        Parameters
        ----------
        M_new
                the number of features to use in the model

        '''
        self.M = M_new

    def laplace(self):
        '''
        We take the original magnetization, pad it with zeros on all sides,
        and then calculate the Laplacian by taking the
        difference between the magnetization at the current point and
        the average of the magnetization at the neighboring points.
        Note that the boundary condition of the laplace is the
        standard Neumann boundary condition.
        Note that the partial derivatives have second-order accuracy.
        Returns
        -------
                The Laplacian of the magnetization.

        '''
        m_laplace = np.zeros_like(self.get_m())
        m_pad = np.pad(self.get_m(), ((1, 1), (1, 1), (1, 1), (0, 0)), 'edge')

        m_laplace += (m_pad[0:-2, 1:-1, 1:-1] + m_pad[2:,
                      1:-1, 1:-1] - 2 * m_pad[1:-1, 1:-1, 1:-1]) / dx ** 2
        m_laplace += (m_pad[1:-1, 0:-2, 1:-1] + m_pad[1:-1,
                      2:, 1:-1] - 2 * m_pad[1:-1, 1:-1, 1:-1]) / dy ** 2
        m_laplace += (m_pad[1:-1, 1:-1, 0:-2] + m_pad[1:-1,
                      1:-1, 2:] - 2 * m_pad[1:-1, 1:-1, 1:-1]) / dz ** 2

        return m_laplace

    def curl(self):
        '''
        We take the curl of the magnetization by taking the difference
        of the partial derivatives of the magnetization with
        respect to the x, y, and z directions.
        Note that the boundary condition of the curl is
        Dirichlet boundary condition.
        Note that the partial derivatives have second-order accuracy.

        Returns
        -------
                The curl of the magnetization.

        '''
        curl = np.zeros_like(self.get_m())

        dzdy = np.zeros((nx, ny, nz))
        dydz = np.zeros((nx, ny, nz))
        dxdz = np.zeros((nx, ny, nz))
        dzdx = np.zeros((nx, ny, nz))
        dxdy = np.zeros((nx, ny, nz))
        dydx = np.zeros((nx, ny, nz))

        m_pad = np.pad(
            self.get_m(), ((1, 1), (1, 1), (1, 1), (0, 0)),
            'constant', constant_values=(
                (0, 0), (0, 0), (0, 0), (0, 0)))

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
