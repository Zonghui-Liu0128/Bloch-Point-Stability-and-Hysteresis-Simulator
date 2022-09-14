import discretisedfield as df
import micromagneticmodel as mm
from src.Mean_field import *
from matplotlib import pyplot as plt
import numpy as np

mu0 = 4 * np.pi * 1e-7
B = -0.1
D = {'bottom': -1.58e-3, 'top': 1.58e-3}
A = 8.78e-12
Ms = 3.84e5


def Initialize(M_array):
    '''
    It creates a mesh, a system, and a field, and then
    initialize the system with a uniform magnetisation
    configuration in +z( or −z )direction.

    Parameters
    ----------
    M_array
        the array of the magnetization

    Returns
    -------
        The magnetization and the mesh.

    '''
    region = df.Region(p1=(0, 0, 0), p2=(150e-9, 150e-9, 30e-9))
    mesh = df.Mesh(region=region, n=(30, 30, 6),
                   subregions={'bottom': df.Region(p1=(0, 0, 0),
                                                   p2=(150e-9, 150e-9, 20e-9)),
                               'top': df.Region(p1=(0, 0, 20e-9),
                                                p2=(150e-9, 150e-9, 30e-9))})
    m = df.Field(mesh, dim=3, value=[0, 0, 1], norm=Ms)
    system = mm.System(name='Mean_field_model')
    system.m = m
    system.energy = (mm.Exchange(A=8.78e-12) +
                     mm.Zeeman(H=(0, 0, B / mm.consts.mu0)) +
                     mm.DMI(D=D, crystalclass='T'))
    m_m = M(system.m.array)
    return m_m, mesh


def Simulator(M_ini_array):
    '''It takes an initial magnetization array as input,
    and returns the final magnetization array

    Parameters
    ----------
    M_ini_array
        the initial magnetization array

    Returns
    -------
        The final magnetization state of the system.

    '''
    m_m, mesh = Initialize(M_ini_array)
    final_status, final_energy = Mean_field_driver(m_m)
    M_final = final_status.get_M()
    m_z = np.average(M_final[..., 2]) / Ms
    print("m_z: ", m_z)
    m = df.Field(mesh, dim=3, value=M_final)
    m.plane('z').mpl()
    plt.show()
    # plt.savefig('T->infinity_Z.pdf', format='pdf')
    m.plane(y=75e-9).mpl()
    # plt.savefig('T->infinity_Y.pdf', format='pdf')
    plt.show()
    m.plane(x=75e-9).mpl()
    # plt.savefig('T->infinity_X.pdf', format='pdf')
    plt.show()
    return M_final


if __name__ == '__main__':
    # data = np.load('M_n_9.npz')
    # M_n_10 = Simulator(data['arr_0'])
    # np.savez('M_n_10', M_n_10)
    Simulator(None)
