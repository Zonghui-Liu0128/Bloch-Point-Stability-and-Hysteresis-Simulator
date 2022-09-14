from src.Energy_Term import *
import math
import numpy as np

mu0 = 4 * np.pi * 1e-7
B = -0.1
D = {'bottom': -1.58e-3, 'top': 1.58e-3}
panel = 4
A = 8.78e-12
K_b = 1.380649e-23

Ms = 3.84e5
nx, ny, nz = (30, 30, 6)
dx, dy, dz = (5e-9, 5e-9, 5e-9)
max_iteration_Mean = 6000
lambda_convergence = 0.005
T = 0
tol = 1e-5
degree = 107


def Langevin(x):
    '''
    The implement of Langevin function

    Parameters
    ----------
    x
        the input to the Langevin function

    Returns
    -------
        the value of the Langevin function.

    '''
    return (1 / np.tanh(x)) - 1 / x


def cal_effective_fields(m):
    '''
    It calculates the effective field and energy of
    a given magnetization configuration.

    Parameters
    ----------
        the magnetization(m)

    Returns
    -------
        the effective field(Heff), the total energy(E)

    '''
    exchange = Exchange(A=A)
    H = np.array([0, 0, B / mu0])
    zeeman = Zeeman(H=H)
    dmi = DMI(D=D, panel=panel)

    Heff_ex = exchange.effective_field(m)
    E_ex = exchange.energy(m)
    Heff_z = zeeman.effective_field(m)
    E_z = zeeman.energy(m)
    Heff_dmi = dmi.effective_field(m)
    E_dmi = dmi.energy(m)
    Heff = Heff_ex + Heff_z + Heff_dmi
    E = E_ex + E_z + E_dmi
    return Heff, E


def end_flag_Mean(M_array, Heff, current_iter=0):
    '''
    To indicate whether the iteration should stop using
    max iterations and Brown's condition.

    Parameters
    ----------
    M_array
        the magnetization array
    Heff
        The effective field.
    current_iter, optional
        the current iteration number

    Returns
    -------
        The return value of this function is a tuple
        of two elements. The first element is a boolean
        value to indicate whether to stop the iteration.
        The flag is True if the stopping criteria is met(m x h_eff = 0),
        and False otherwise. The second element is the condition term,
        which is the cross product of the unit vector of the M_array
        and the Heff.

    '''

    condition_term = np.zeros_like(M_array)
    if (np.allclose(M_array, 0)):
        return True, condition_term
    else:
        m_array = M_array / \
            np.expand_dims(np.linalg.norm(M_array, axis=3), axis=3)
        heff = Heff / np.expand_dims(np.linalg.norm(Heff, axis=3), axis=3)
        condition_term = np.cross(m_array, heff)
        condition_term_norm = np.expand_dims(
            np.linalg.norm(condition_term, axis=3), axis=3)
        condition_term_norm_max = np.max(condition_term_norm)

        if current_iter > max_iteration_Mean or np.allclose(
                condition_term_norm_max, 0, atol=tol):
            return True, condition_term
        else:
            return False, condition_term


def end_flag_Mean_BP(M_array, Heff, current_iter=0):
    '''
    To indicate whether the iteration should stop using max iterations
    and the physical definition of the Bloch point.
    When the magnetic field angle at the interface between
    the two chiral materials is greater than 100 degrees,
    the Bloch point is considered to appear. The angle can be defined manually.

    Parameters
    ----------
    M_array
        the magnetization array
    Heff
        the effective field
    current_iter, optional
        the current iteration number

    Returns
    -------
        a boolean value and a cross product array.

    '''
    cross_product = np.zeros_like(M_array)
    if (np.allclose(M_array, 0)):
        return True, cross_product
    else:
        m_array = M_array / \
            np.expand_dims(np.linalg.norm(M_array, axis=3), axis=3)
        heff = Heff / np.expand_dims(np.linalg.norm(Heff, axis=3), axis=3)
        cross_product = np.cross(m_array, heff)
        boundary_b = m_array[:, :, panel - 1]
        boundary_u = m_array[:, :, panel]
        tol_BP = np.cos(math.radians(degree))
        gate = np.sum(np.multiply(boundary_b, boundary_u), axis=2)
        if (np.any(gate < tol_BP) or current_iter > max_iteration_Mean):
            return True, cross_product
        else:
            return False, cross_product


def update_effective_fields(M_old_array, Ms_old_array, Heff):
    '''
    It updates the magnetization of each cell in the mesh at once.
    The update can be devided into two parts:
    1) the change of the length
    2) the change of the direction
    and then combine the two changes and get the new magnetization.

    Parameters
    ----------
    M_old_array
        the old magnetization
    Ms_old_array
        the saturation magnetization of the material
    Heff
        the effective field

    Returns
    -------
        The new magnetization.

    '''
    Heff_norm = np.expand_dims(np.linalg.norm(Heff, axis=3), axis=3)
    if T == 0:
        res_Langevin = 1
    else:
        # We introduce the temperature here.
        beta = 1 / (T * K_b)
        x = beta * mu0 * Heff_norm
        res_Langevin = Langevin(x)

    if (np.allclose(Heff_norm, 0)):
        M_new_array = M_old_array
    else:
        # Change the length of the magnetization
        M_len_array = np.multiply(
            Ms_old_array, np.multiply(
                res_Langevin, (Heff / Heff_norm)))
        # Change the direction of the magnetization
        M_dir_array = M_old_array + lambda_convergence * \
            (M_len_array - M_old_array)
        # Get the new direction
        uni_vec = M_dir_array / \
            np.expand_dims(np.linalg.norm(M_dir_array, axis=3), axis=3)
        # Combine the change of length and that of direction to get the new
        # magnetization
        M_new_array = uni_vec * \
            np.expand_dims(np.linalg.norm(M_len_array, axis=3), axis=3)
    return M_new_array


def Mean_field_driver(m):
    '''
    The Mean-field driver: The function takes the magnetization as
    an input and returns the relaxed system's magnetization
    and the total energy after using the Mean-field driver.

    Parameters
    ----------
    m
        the magnetization object

    Returns
    -------
        The magnetization and the energy of the system.

    '''
    flag_stop = False
    cnt_iter = 0
    E = 0
    while not flag_stop:
        Heff, E = cal_effective_fields(m)
        M_old_array = m.get_M()
        Ms_old_array = m.get_Ms()
        M_new_array = update_effective_fields(M_old_array, Ms_old_array, Heff)
        m.set_M(M_new_array)
        flag_stop, cross_product = end_flag_Mean(M_new_array, Heff, cnt_iter)
        cnt_iter += 1
        # To follow the progress of the iteration, we print the number of
        # iterations every 100 iterations.
        if (cnt_iter % 100 == 0):
            print("Iteration: ", cnt_iter)

    print("Number of iteration: ", cnt_iter)
    return m, E
