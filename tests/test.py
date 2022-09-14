import sys
sys.path.append(".")
import discretisedfield as df  # noqa: E402
import micromagneticmodel as mm  # noqa: E402
import oommfc as oc  # noqa: E402
import numpy as np  # noqa: E402
from src.Energy_Term import Zeeman, Exchange, DMI, M  # noqa: E402
from src.Simulator import Simulator, Initialize  # noqa: E402


# Defining the parameters of the system.
A = 8.78e-12
panel = 4
D = {'bottom': -1.58e-3, 'top': 1.58e-3}
mu0 = 4 * np.pi * 1e-7
B = -0.1
Ms = 3.84e5

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


class TestEnergy:
    def test_effective_exchange_zeeman(self):
        """
        It tests that the effective field of the Zeeman
        energy term is computed correctly
        """
        H = np.array([0, 0, B / mu0])
        zeeman = Zeeman(H=H)
        Heff_z = zeeman.effective_field(m_m)
        Heff_z_ubermag = oc.compute(system.energy.zeeman.
                                    effective_field, system).array
        assert np.allclose(Heff_z, Heff_z_ubermag)

    def test_effective_exchange_exchange(self):
        """
        It tests that the effective field computed by the exchange
        energy object is the same as the one computed by the
        ubermag
        """
        exchange = Exchange(A=A)
        Heff_ex = exchange.effective_field(m_m)
        Heff_ex_ubermag = oc.compute(system.energy.
                                     exchange.effective_field, system).array
        assert np.allclose(Heff_ex, Heff_ex_ubermag)

    def test_effective_exchange_dmi(self):
        """
        It tests that the effective field computed by the DMI
        object is the same as the one computed by Ubermag
        """
        dmi = DMI(D=D, panel=panel)
        Heff_dmi = dmi.effective_field(m_m)
        Heff_dmi_ubermag = oc.compute(system.energy.
                                      dmi.effective_field, system).array
        assert np.allclose(Heff_dmi, Heff_dmi_ubermag)

    def test_density_zeeman(self):
        """
        > This function tests the energy density of the Zeeman energy term
        """
        H = np.array([0, 0, B / mu0])
        zeeman = Zeeman(H=H)
        density_z = zeeman.energy_density(m_m)
        density_z_ubermag = oc.compute(system.energy.
                                       zeeman.density, system).array
        assert np.allclose(density_z, density_z_ubermag)

    def test_density_exchange(self):
        """
        It tests whether the exchange energy density computed by
        the `Exchange` class is the same as the one computed by
        the `ubermag` package
        """
        exchange = Exchange(A=A)
        density_ex = exchange.energy_density(m_m)
        density_ex_ubermag = oc.compute(system.energy.
                                        exchange.density, system).array
        assert np.allclose(density_ex, density_ex_ubermag)

    def test_density_dmi(self):
        """
        It tests whether the energy density of the DMI interaction computed
        by the `DMI` class is the same as the energy density of the DMI
        interaction computed by the `ubermag` package
        """
        dmi = DMI(D=D, panel=panel)
        density_dmi = dmi.energy_density(m_m)
        density_dmi_ubermag = oc.compute(system.energy.
                                         dmi.density, system).array
        assert np.allclose(density_dmi, density_dmi_ubermag)

    def test_energy_zeeman(self):
        """
        It tests whether the energy of a Zeeman
        interaction is computed correctly
        """
        H = np.array([0, 0, B / mu0])
        zeeman = Zeeman(H=H)
        energy_z = zeeman.energy(m_m)
        energy_z_ubermag = oc.compute(system.energy.zeeman.energy, system)
        assert np.isclose(energy_z, energy_z_ubermag)

    def test_energy_exchange(self):
        """
        It tests whether the energy of the exchange interaction computed
        by the `Exchange` class is the same as the energy
        computed by the `ubermag` package
        """
        exchange = Exchange(A=A)
        energy_ex = exchange.energy(m_m)
        energy_ex_ubermag = oc.compute(system.energy.exchange.energy, system)
        assert np.isclose(energy_ex, energy_ex_ubermag)

    def test_energy_dmi(self):
        '''
        It tests whether the energy of the DMI interaction computed by
        the `DMI` class is the same as the energy computed by
        the `ubermag` package

        '''
        dmi = DMI(D=D, panel=panel)
        energy_dmi = dmi.energy(m_m)
        energy_dmi_ubermag = oc.compute(system.energy.
                                        dmi.energy, system)
        assert np.isclose(energy_dmi, energy_dmi_ubermag)
