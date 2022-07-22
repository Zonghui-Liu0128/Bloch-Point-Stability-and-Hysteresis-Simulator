from EnergyTerm import *
import numpy as np
import discretisedfield as df
import micromagneticmodel as mm
import oommfc as oc
from Mean_field import *

# Some important hyperparameters
Ms = 3.84e5
A = 8.78e-12
mu0 = 4 * np.pi * 1e-7
D = 1.58e-3
B = 0.1


# Print the effective fields' values to the txt files and compare it with Ubermag's values
def print_effective_fields(path_Ubermag=None, path_my=None, Heff=None, Heff_my=None):
    file1 = open(path_Ubermag, 'w', encoding='UTF-8')
    for i in range(len(Heff.array)):
        file1.write(str(Heff.array[i]) + '\n')
    file1.close()

    file2 = open(path_my, 'w', encoding='UTF-8')
    for i in range(len(Heff_my)):
        file2.write(str(Heff_my[i]) + '\n')
    file2.close()


# Print the densities' values and total energies to the txt files and compare it with Ubermag's values
def print_density_energy(path_Ubermag=None, path_my=None, w=None, w_my=None, E=0, E_my=0):
    file1 = open(path_Ubermag, 'w', encoding='UTF-8')
    for i in range(len(w.array)):
        file1.write(str(w.array[i]) + '\n')
    file1.write(str(E) + '\n')
    file1.close()

    file2 = open(path_my, 'w', encoding='UTF-8')
    for i in range(len(w_my)):
        file2.write(str(w_my[i]) + '\n')
    file2.write(str(E_my) + '\n')
    file2.close()


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


# Initialize the system
region = df.Region(p1=(0, 0, 0), p2=(50e-9, 50e-9, 50e-9))
mesh = df.Mesh(region=region, cell=(5e-9, 5e-9, 5e-9), bc="neumann")
m = df.Field(mesh, dim=3, value=M_m, norm=Ms)
system = mm.System(name='energy_term_testing')
system.m = m
m_m = M(system.m.array / Ms)
system.energy = (mm.DMI(D=1.58e-3, crystalclass='T') +
                 mm.Exchange(A=8.78e-12) +
                 mm.Zeeman(H=(0, 0, B / mm.consts.mu0)))

''' Test for the exchange energy term '''
# Heff_ex = oc.compute(system.energy.exchange.effective_field, system)
# w_ex = oc.compute(system.energy.exchange.density, system)
# E_ex = oc.compute(system.energy.exchange.energy, system)
#
# exchange = Exchange(A=A)
# Heff_ex_my = exchange.effective_field(m_m)
# w_ex_my = exchange.energy_density(m_m)
# E_ex_my = exchange.energy(m_m)
#
# # Compare
# print_effective_fields('/Users/lzh/PycharmProjects/Mean_field/Heff_ex_Ubermag.txt',
#                        '/Users/lzh/PycharmProjects/Mean_field/Heff_ex_my.txt', Heff_ex, Heff_ex_my)
# print_density_energy('/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_ex_Ubermag.txt',
#                      '/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_ex_my.txt', w_ex, w_ex_my, E_ex, E_ex_my)
#
''' Test for the zeeman energy term '''
# Heff_z = oc.compute(system.energy.zeeman.effective_field, system)
# w_z = oc.compute(system.energy.zeeman.density, system)
# E_z = oc.compute(system.energy.zeeman.energy, system)
#
# H = np.array([0, 0, B / mu0])
# zeeman = Zeeman(H=H)
# Heff_z_my = zeeman.effective_field(m_m)
# w_z_my = zeeman.energy_density(m_m)
# E_z_my = zeeman.energy(m_m)

# Compare
# print_effective_fields('/Users/lzh/PycharmProjects/Mean_field/Heff_z_Ubermag.txt',
#                        '/Users/lzh/PycharmProjects/Mean_field/Heff_z_my.txt', Heff_z, Heff_z_my)
# print_density_energy('/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_z_Ubermag.txt',
#                      '/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_z_my.txt', w_z, w_z_my, E_z, E_z_my)
''' Test for the dmi energy term '''
# Heff_dmi = oc.compute(system.energy.dmi.effective_field, system)
# w_dmi = oc.compute(system.energy.dmi.density, system)
# E_dmi = oc.compute(system.energy.dmi.energy, system)
#
# dmi = DMI(D)
# Heff_dmi_my = dmi.effective_field(m_m)
# w_dmi_my = dmi.energy_density(m_m)
# E_dmi_my = dmi.energy(m_m)
#
# # Compare
# print_effective_fields('/Users/lzh/PycharmProjects/Mean_field/Heff_dmi_Ubermag.txt',
#                        '/Users/lzh/PycharmProjects/Mean_field/Heff_dmi_my.txt', Heff_dmi, Heff_dmi_my)
# print_density_energy('/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_dmi_Ubermag.txt',
#                      '/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_dmi_my.txt', w_dmi, w_dmi_my, E_dmi, E_dmi_my)
''' Test for the three energy term(exchange, dmi and zeeman) '''
# Heff = oc.compute(system.energy.effective_field, system)
# w = oc.compute(system.energy.density, system)
# E = oc.compute(system.energy.energy, system)
#
# exchange = Exchange(A=A)
# H = np.array([0, 0, B / mu0])
# zeeman = Zeeman(H=H)
# dmi = DMI(D)
#
# Heff_ex_my = exchange.effective_field(m_m)
# w_ex_my = exchange.energy_density(m_m)
# E_ex_my = exchange.energy(m_m)
#
# Heff_z_my = zeeman.effective_field(m_m)
# w_z_my = zeeman.energy_density(m_m)
# E_z_my = zeeman.energy(m_m)
#
# Heff_dmi_my = dmi.effective_field(m_m)
# w_dmi_my = dmi.energy_density(m_m)
# E_dmi_my = dmi.energy(m_m)
#
# Heff_my = Heff_ex_my + Heff_z_my + Heff_dmi_my
# w_my = w_ex_my + w_z_my + w_dmi_my
# E_my = E_ex_my + E_z_my + E_dmi_my
#
# min_driver = Min_Driver()
# Heff_my = min_driver.cal_effective_fields(m_m)
# print_effective_fields('/Users/lzh/PycharmProjects/Mean_field/Heff_Ubermag.txt',
#                        '/Users/lzh/PycharmProjects/Mean_field/Heff_my.txt', Heff, Heff_my)
# print_density_energy('/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_Ubermag.txt',
#                      '/Users/lzh/PycharmProjects/Mean_field/DensityEnergy_my.txt', w, w_my, E, E_my)

''' Test for the Mean field model '''
E_ini = oc.compute(system.energy.energy, system)
print("The total energy of the initialized system: ", E_ini)
print("**" * 15)

md = oc.MinDriver()
md.drive(system)
final_moments_array = system.m.array / Ms
m_min_driver = M(final_moments_array)
E_min_driver = oc.compute(system.energy.energy, system)
print("The min total energy calculated by Ubermag: ", E_min_driver)


print("**" * 15)
min_driver = Min_Driver()
final_moments_my = min_driver.Mean_field_driver(m_m)
exchange = Exchange(A=A)
H = np.array([0, 0, B / mu0])
zeeman = Zeeman(H=H)
dmi = DMI(D)
E_ex_my = exchange.energy(final_moments_my)
E_z_my = zeeman.energy(final_moments_my)
E_dmi_my = dmi.energy(final_moments_my)
E_my = E_ex_my + E_z_my + E_dmi_my
print("The min total energy calculated by Mean Field: ", E_my)

# print_effective_fields('/Users/lzh/PycharmProjects/Mean_field/MinDriver.txt',
#                        '/Users/lzh/PycharmProjects/Mean_field/Mean_field.txt', final_moments, final_moments_my)
