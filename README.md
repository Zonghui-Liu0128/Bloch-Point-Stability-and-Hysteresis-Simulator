![BlochPoint](https://github.com/ese-msc-2021/irp-zl1021/blob/main/figs/BP_project.png)

# A Three-dimensional Bloch Point Stability and Hysteresis Simulator


The 3D Bloch point stability and hysteresis simulator is implemented with Python 3. The Mean-field algorithm is used in the simulator to relax the object system to the equilibrium state. Using the simulator, two types of Bloch points could be found in the cubic B20 FeGe system with two chiralities when the temperature $T = 0\mathrm{K}$. Using the simulator, we can also simulate the hysteretic behaviour of the system. What's more, the simulator also introduces the effect of temperature on the system. The figures shown above indicate the two types of Bloch points(H-H and T-T).

This project is part of Zonghui Liu's IRP of the MSc. ACSE at Imperial College London. More details about the Mean-field algorithm and the description of the simulator can be found in [Zonghui Liu's final report](https://github.com/ese-msc-2021/irp-zl1021/blob/main/reports/zl1021-final-report.pdf).

## Author

Zonghui Liu

If you have any problems, please feel free to submit an issue or contact me via email <zl1021@ic.ac.uk>.

## Installation
The development version can be installed by cloning the repository:

```
git clone https://github.com/ese-msc-2021/irp-zl1021.git
```

## Dependencies

- Python 3.8+
- NumPy 1.23.2
- Matplotlib 3.5.3
- discretisedfield 0.64.0
- micromagneticmodel 0.63.0
- oommfc 0.63.0


**Note**: The open source package Ubermag was used in the initialization, plotting, and testing of this project, therefore, more details about the installation and usage of Ubermag could be found [here](https://ubermag.github.io/).

## Structures

As the implementation of this project is based on Object-Oriented Python, the organisation of the code is given from the Unified Modeling Language diagram:

![UML](https://github.com/ese-msc-2021/irp-zl1021/blob/main/figs/UML.jpeg)

The next section describes the classes and functions contained in each file:

```
├── code
   ├── Energy_Term.py
   		├── class EnergyTerm: the interface, indicates the relationships of 
   		effective field, density and energy.
   		├── class Exchange(EnergyTerm): it calculates the effective field, 
   		density and energy for exchange energy term.
   		├── class Zeeman(EnergyTerm): it calculates the effective field, 
   		density and energy for zeeman energy term.
   		├── class DMI(EnergyTerm): it calculates the effective field, 
   		density and energy for DMI energy term.
   		├── class M:
   			├── get_M()
   			├── get_Ms()
   			├── get_m()
   			├── set_M()
   			├── laplace(): used for calculating the effective field of exchange 
   			energy
   			├── curl(): used for calculating the effective field of DMI energy 
   			term
   ├── Mean_field.py
   		├── Mean_field_driver(): the main function of Mean-field algorithm
   		├── update_effective_fields(): to update Heff at each iteration
   		├── end_flag_Mean(): uses the max iterations and the Brown's condition to 
   		end the iterations
   		├── end_flag_Mean_BP(): uses the max iterations, Brown's condition and 
   		the angle of magnetization at the interface to control the iterations
   		├── cal_effective_fields(): calculates the effective fields of the 
   		system. It is easy to add other energy term, such as the Demagnetization 
   		and anisotropy.
   		├── Langevin(): to introduce the effect of temperature
   ├── Simulator.py
   		├── Initialize(): to initialize the system using Ubermag
   		├── main(): the entrance of the project
   ├── test.py
   		├── class TestEnergy: Test the calculation of effective field, density 
   		and energy for exchange, zeeman and dmi, and compared it with Ubermag's 
   		results.
```

## Usage

### Change the material
![disk](https://github.com/ese-msc-2021/irp-zl1021/blob/main/figs/disk.png)

The object system is shown above.

To change the material, the exchange constant `A`, the DMI constant `D` and the saturation magnetisation `Ms` should be change accordingly. They are defined at the top of the four files. The examples of parameters of cubic B20 FeGe is given by:

```
A = 8.78pJm^1
|D| = 1.58mJm^−2
Ms = 384kAm^−1
```

### Change the size, shape or number of lattices

The object system is seen as a continuous matter, the system needs to be discretized as a number of lattices using the finite difference method in order to use computer to compute. To change the size, shape and number of lattices, the `Initialize()` function and the tuple `(nx, ny, nz)` and `(dx, dy, dz)` need to be changed accordingly. There is an example set of parameters:

```
mesh = df.Mesh(region=region, n=(60, 60, 6),
                   subregions={'bottom': df.Region(p1=(0, 0, 0), p2=(150e-9, 150e-9, 20e-9)),
                               'top': df.Region(p1=(0, 0, 20e-9), p2=(150e-9, 150e-9, 30e-9))})
                               
nx, ny, nz = (60, 60, 6)
dx, dy, dz = (2.5e-9, 2.5e-9, 5e-9)
```

### Change the temperature

To study Bloch points' stability at different temperature, the experimental temperature `T` should be changed. The temperature is defined at the top of the `Mean_field.py` file. There is an example: `T = 0`. The temperature is introduced into the Mean-field algorithm to calculate the parameter $\beta$, then to update the length of magnetizations, and finally to compute the new magnetizations.

### Change the parameters of Mean-field model

All parameters used in Mean-field model are defined at the top of the `Mean_field.py` file. The example is given by:

```
max_iteration_Mean = 8000
lambda_convergence = 0.005
T = 0
tol = 1e-5
degree = 107
```

### Study the hysteretic behaviour of the system
![Hysteresis](https://github.com/ese-msc-2021/irp-zl1021/blob/main/figs/Hysteresis.png)

In order to study the hysteretic behaviour of the system, we need to use the final magnetizations after Mean-field Driver using the last $\mu_0H$. To do that, the project provides the I/O operations from/to `.npz` files. In the `Simulator.py` file, the three lines in the `main` function should be set:

```
data = np.load('M_n_9.npz')	# The last final magnetizations
M_n_10 = Simulator(data['arr_0'])		# Use the Mean-field model to relax
np.savez('M_n_10', M_n_10)		# Save the final magnetizations
```
Note that the final magnetization files are saved at the same path of `Simulator.py` file.

## Some interesting results and their parameters

### Some important parameters to find stable Bloch points

| Type of BP  | No.Cells    | Size of cell(nm)| $\mu_0H(T)$ | Temperature(K)| Iter rate |
| ----------- | ----------- | -------------------------- | ----------- |--------------------------|-----------|
| Head-to-Head| (30, 30, 6) |         (5, 5, 5)          |    -0.1     |      0                   |   0.005   |
| Tail-to-Tail| (30, 30, 6) |         (5, 5, 5)          |     0.1     |      0                   |   0.005   |


## To Physicists

Here are some important settings that I hope will help you in your work:

- The boundary condition for the exchange energy term: standard **Neumann** with 2nd order
- The boundary condition for the dmi energy term: **Dirichlet**
- To stop the iterations: the max iterations and the cross product(m x heff) are used to determine whether the algorithm convergent. However, when the Bloch point appears at the interface, since the magnetizations are **not continuous** at the Bloch point, the cross product is large. To stop the iterations when there is a Bloch point, the angle of magnetizations at the interface is added to the stopping criterias. When the angles of magnetizations at the interface are all greater than 100 degree, we believe the Bloch point appears. The angle could be set to a different degree.


## Acknowledge

Thanks to Yang Bai, Sijia Chen, Haoyu Wu, Xiaoyi Yu and Yuhang Zhang for their assistance. Special thanks to Dr. Marijan for his guidance and supervision. 😊