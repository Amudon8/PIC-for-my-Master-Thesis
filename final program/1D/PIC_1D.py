import numpy as np
import random
import scipy.interpolate as sp
import csv
import matplotlib.pyplot as plt

new_file = open('electric_field.csv', 'w')
cvs_writer = csv.writer(new_file, delimiter=',')
cvs_writer.writerow(np.linspace(0, 1, 101))
new_file.close()


time_step = float(input("Enter the time step: "))
#length of system
length = float(input("Enter the length of the system: "))
no_grid = int(input("Enter the no. of grid: "))
#defining grid
grid_spacing = length/no_grid
grid = np.arange(0, length+grid_spacing, grid_spacing)

no_particle = int(input("Enter the no. of particle: "))
elec_pos = np.linspace(0, length, no_particle)
ion_pos = np.linspace(0, length, no_particle)

#perturbation
pert_length = []
for i in range(0, no_particle):
    a = random.uniform(-1, 1)
    pert_length.append(a)
elec_pos = elec_pos + pert_length

#defining initial random velocity
vel_therm = float(input("Enter the value of thermal velocity: "))
vel = []
for i in range(0, no_particle):
    val = random.uniform(-vel_therm, vel_therm)
    vel.append(val)
vel_step = int(input("Enter the no. of segment of velocity: "))
vel_dis, edge = np.histogram(vel, bins=vel_step)
plt.subplot(1, 2, 1)
plt.title('Before simulation')
plt.hist(vel, bins=50, histtype='step', density=True)
plt.xlabel("Velocity in arb. unit")
plt.ylabel("Normalised No. of Particles")
print("Initial velocity distribution")
print(vel_dis)

ion_vel = [0]*no_particle

particle_charge = -1
ion_charge = 0.01
#main loop
for j in range(0, 5000):
    print(j)
    for i in range(0, no_particle):
        if elec_pos[i] > length:
            a = elec_pos[i] - length
            elec_pos[i] = length - a
            vel[i] = -vel[i]
        elif elec_pos[i] < 0:
            elec_pos[i] = -elec_pos[i]
            vel[i] = -vel[i]
        else:
            elec_pos[i] = elec_pos[i]

        if ion_pos[i] > length:
            a = ion_pos[i] - length
            ion_pos[i] = length - a
            ion_vel[i] = -ion_vel[i]
        elif ion_pos[i] < 0:
            ion_pos[i] = -ion_pos[i]
            ion_vel[i] = -ion_vel[i]
        else:
            ion_pos[i] = ion_pos[i]
    pert_dis, edge = np.histogram(elec_pos, bins=no_grid)
    ion_dis, ion_edge = np.histogram(ion_pos, bins=no_grid)
    ion_count = np.sum(ion_dis)
    count = np.sum(pert_dis)
    if count == no_particle and ion_count == no_particle:
        electron_charge_den = (pert_dis - ion_dis) * particle_charge
        charge_density_node = [0] * (no_grid + 1)
        for i in range(0, no_grid + 1):
            if i == 0:
                charge_density_node[i] = electron_charge_den[i]  #this is to make the flux xero
            elif i == no_grid:
                charge_density_node[i] = electron_charge_den[i-1]
            else:
                charge_density_node[i] = (electron_charge_den[i - 1] + electron_charge_den[i]) * 0.5
        electric_field = np.cumsum(charge_density_node) * grid_spacing
        inter_func = sp.interp1d(grid, electric_field, 'cubic')

        #equation of motion
        accel = particle_charge * inter_func(elec_pos)
        ion_accel = ion_charge * inter_func(ion_pos)

        vel = vel + accel * time_step
        ion_vel = ion_vel + ion_accel * time_step

        elec_pos = elec_pos + vel * time_step
        ion_pos = ion_pos + ion_vel * time_step

        x = np.linspace(0, 1, 101)
        new_file = open('electric_field.csv', 'a')
        cvs_writer = csv.writer(new_file, delimiter=',')
        cvs_writer.writerows([inter_func(x)])
        new_file.close()
    else:
        print("not working")

plt.subplot(1, 2, 2)
plt.title('After simulation')
plt.hist(vel, bins=50, histtype='step', density=True)
plt.xlabel("Velocity in arb. unit")
plt.ylabel("Normalised No. of Particles")

plt.show()

