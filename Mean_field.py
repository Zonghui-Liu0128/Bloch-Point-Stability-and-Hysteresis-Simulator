import numpy as np
from EnergyTerm import *
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc

# Some important hyperparameters
mu0 = 4 * np.pi * 1e-7
Ms = 3.84e5
nx, ny, nz = (10, 10, 10)
dx, dy, dz = (5e-9, 5e-9, 5e-9)
A = 8.78e-12
D = 1.58e-3
B = 0.1
T = 298
K_b = 1.380649e-23
lambda_convergence = 0.5
max_iteration_Mean = 10000
tol = 1e-4


# Test case
def M_x(x, y, z):
    return 3 * x ** 2 - 2 * y ** 3 - z


def M_y(x, y, z):
    return 5 * x - 3 * y + z ** 2


def M_z(x, y, z):
    return -3 * x - y ** 3 + z ** 2


def M_m(point):
    x, y, z = point
    return M_x(x, y, z), M_y(x, y, z), M_z(x, y, z)




class Min_Driver:
    def Langevin(self, x):
        return (1 / np.tanh(x)) - 1 / x

    # Tested
    def cal_effective_fields(self, m):
        exchange = Exchange(A=A)
        H = np.array([0, 0, B / mu0])
        zeeman = Zeeman(H=H)
        dmi = DMI(D)

        Heff_ex_my = exchange.effective_field(m)
        Heff_z_my = zeeman.effective_field(m)
        Heff_dmi_my = dmi.effective_field(m)
        Heff = Heff_ex_my + Heff_z_my + Heff_dmi_my
        return Heff

    def update_effective_fields(self, m_old, Heff):
        beta = 1 / (T * K_b)  # Introduce the temperature
        Heff_norm = np.expand_dims(np.linalg.norm(Heff, axis=3), axis=3)
        x = beta * Ms * mu0 * Heff_norm
        m_new_array = self.Langevin(x) * Heff / Heff_norm
        m_old_array = m_old.field()
        m_new_array = m_old_array + lambda_convergence * (m_new_array - m_old_array)
        return m_new_array

    def end_flag_Mean(self, m, Heff, current_iter=0):
        m_array = m.field() * Ms    # it should be m or M ????
        # Stop when current iteration > max iteration of Mean_field model,
        # or M x Heff = 0
        if current_iter > max_iteration_Mean or np.allclose(np.zeros((nx, ny, nz, 3)), np.cross(m_array, Heff), atol=tol):
            return True
        else:
            return False

    def Mean_field_driver(self, m):
        flag_stop = False  # Used to determine if iteration should be stopped
        cnt_iter = 0  # the iteration counter
        while not flag_stop:
            # Compute the effective fields for all cubes at once
            Heff = self.cal_effective_fields(m)
            # Update the magnetic moment
            m_new_array = self.update_effective_fields(m, Heff)
            m.set_field(m_new_array)
            # Check end or not
            flag_stop = self.end_flag_Mean(m, Heff, cnt_iter)
            cnt_iter += 1

        print("Number of iteration: ", cnt_iter)
        return m

