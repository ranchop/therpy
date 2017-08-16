# misc functions

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate as tbl
import pandas as pd
import time as time_lib

from . import constants

###############
# Lithium Imaging
cst = constants.cst(sigmaf=0.5)


class LithiumImagingSimulator:
    '''
    Setting up:
        z, dz --from-- z_step, z_max
        t, dt --from-- t_step, t_max
        I, n: for 2D visualization
    Initial conditions:
        I0 --from-- I0
        n0 --from-- density, col_depth, n_start

        trackers

    Simulation Results:
        If, dI, od, n_od, n_tot, max_disp

    Methods:
        update_database(database_name)

    Private Methods:
        initial_setup
        time_evolve
        sim_results

    '''

    def __init__(self, **kwargs):
        self.var = kwargs
        self.initial_setup()
        self.time_evolve()
        self.sim_results()
        self.info()
        if self.var.get('plot', False):
            self.infoplot()

    def info(self):
        # info dict
        infodict = [['density', self.density],
                    ['intensity', self.I0],
                    ['last od', self.od[-1]]]
        print("=============================================")
        print("Grid size in z {} and in t {}".format(self.z.size, self.t.size))
        print("Keeping track of {} points.".format(len(self.trackers)))
        print("Calculations took {:.2f} seconds.".format(self.elapsed_time))
        print(tbl(infodict))

    def infoplot(self):
        fig1, ax = plt.subplots(ncols=2, figsize=(8, 3))
        clim_I = [self.I[-1, -1], self.I[0, 0]]
        clim_n = [0, np.max(self.n)]
        ax[0].imshow(self.I, cmap='viridis', aspect='auto', origin='lower', clim=clim_I)
        ax[0].set(title='Intensity', xlabel='position [pixels]', ylabel='time [pixels]')
        ax[1].imshow(self.n, cmap='viridis', aspect='auto', origin='lower', clim=clim_n)
        ax[1].set(title='Density', xlabel='position [pixels]', ylabel='time [pixels]')
        fig2, ax = plt.subplots(ncols=3, figsize=(13, 3.5))
        ax[0].plot(self.t * 1e6, self.od)
        ax[0].set(title='Observed OD', xlabel="Imaging Time [$\mu$s]")
        ax[1].plot(self.t * 1e6, self.n_od)
        ax[1].set(title='Atoms: OD only', xlabel="Imaging Time [$\mu$s]")
        ax[2].plot(self.t * 1e6, self.n_tot)
        ax[2].set(title='Atoms: OD + Isat', xlabel="Imaging Time [$\mu$s]")

    def initial_setup(self):
        # Setup z
        z_max = self.var.get('z_max', 200) * 1.0e-6
        self.dz = self.var.get('dz', 0.1) * 1.0e-6
        self.z = np.arange(0.0, z_max + self.dz, self.dz)
        # Setup t
        t_max = self.var.get('t_max', 10) * 1.0e-6
        self.dt = self.var.get('dt', 0.1) * 1.0e-6
        self.t = np.arange(0.0, t_max + self.dt, self.dt)
        # 2D grids
        self.I = np.zeros(shape=(self.t.size, self.z.size))
        self.n = np.zeros(shape=(self.t.size, self.z.size))
        # Initial conditions - Intensity
        self.I0 = self.var.get('I0', 0.01)
        self.I[:, 0] = self.I0
        # Initial conditions - Density
        self.density = self.var.get('density', 1e17)
        self.col_depth = self.var.get('col_depth', 120) * 1e-6
        self.col_density = self.col_depth * self.density
        n_start = self.var.get('n_start', 10) * 1e-6
        self.n0 = np.ones_like(self.z) * self.density
        self.n0[np.logical_or(self.z < n_start, self.z >= n_start + self.col_depth)] = 0
        # trackers
        z0 = self.z[self.n0 > 0]
        n0 = self.n0[self.n0 > 0]
        self.trackers = [particle(n=n0[i], z0=z0[i], v0=0.0, tv=self.t, zv=self.z) for i in range(z0.size)]
        # Tell myself that its NOT ok to write these data to a databse
        self.ok2write = False

    def time_evolve(self):
        # Start timer
        start = time_lib.time()
        # Step through time 0 to end-1
        for ti in range(self.t.size - 1):
            # Find I(z) at current time
            self.I[ti, :] = calc_I(self.I[ti, 0], ti, self.trackers, self.z, self.dz)
            # Evolve trackers to next time step
            for a in self.trackers: a.evolve(ti, self.I[ti, :])
        # Find I(z) for the last time step
        ti = self.t.size - 1
        self.I[ti, :] = calc_I(self.I[ti, 0], ti, self.trackers, self.z, self.dz)
        # Stop timer
        stop = time_lib.time()
        self.elapsed_time = stop - start

    def sim_results(self):
        # calculate n(t,z)
        calc_n(self.n, self.trackers)
        # calculate important stats
        self.od, self.If, self.n_od, self.n_tot = calc_sim_results(self.I, self.n0, self.z, self.t, self.col_density)
        self.dI = 1 - self.If
        self.max_disp = np.array([np.max([a.z[ti] - a.z[0] for a in self.trackers]) for ti in range(self.t.size)]) * 1e6
        # Tell myself that its now ok to write these data to a databse
        self.ok2write = True

    def update_database(self, name=None, new=False, times=(0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
        if name is None:
            print("PLEASE provide a full path to database file.")
            return None
        # Create data for new database
        times = np.array(times)
        density = self.density * np.ones_like(times)
        intensity = self.I0 * np.ones_like(times)
        col_depth = self.col_depth * np.ones_like(times)
        col_density = self.col_density * np.ones_like(times)
        od = np.interp(times, self.t * 1e6, self.od)
        dI = np.interp(times, self.t * 1e6, self.dI)
        n_od = np.interp(times, self.t * 1e6, self.n_od)
        n_tot = np.interp(times, self.t * 1e6, self.n_tot)
        max_disp = np.interp(times, self.t * 1e6, self.max_disp)
        db_dict = dict(time=times, density=density, intensity=intensity,
                       col_depth=col_depth, col_density=col_density,
                       od=od, dI=dI, n_od=n_od, n_tot=n_tot,
                       max_disp=max_disp)
        df_add = pd.DataFrame(db_dict, columns=['density', 'intensity', 'time',
                                                'col_depth', 'col_density',
                                                'od', 'dI', 'n_od', 'n_tot',
                                                'max_disp'])
        if not new:
            # Import current database
            df_read = pd.read_csv(name)
            # Combine databases
            df_total = df_read.append(df_add, ignore_index=True)
        else:
            df_read = pd.DataFrame()
            df_total = df_add
        # Write to the same csv
        if self.ok2write:
            print("Initial databse length {}; Added {} rows. Final length {}".format(df_read.shape[0], df_add.shape[0],
                                                                                     df_total.shape[0]))
            df_total.to_csv(name, index=False)
            self.ok2write = False
        else:
            print("Not allowed to write again. Possible that it has already been written once.")


class particle:
    def __init__(self, n, z0, v0, tv, zv):
        self.n = n
        self.z = np.zeros_like(tv)
        self.z[0] = z0
        self.v = np.zeros_like(tv)
        self.v[0] = v0
        self.zv = zv
        self.dt = tv[1] - tv[0]

    def accel(self, ti, Iz):
        v = self.v[ti]
        I = np.interp(self.z[ti], self.zv, Iz)
        return cst.hbar * cst.k * cst.gamma / 2 * I / (1 + I + (2 * v * cst.k / cst.gamma) ** 2) / cst.mass

    def evolve(self, ti, Iz):
        # Evolve position according to velocity at ti
        self.z[ti + 1] = self.z[ti] + self.v[ti] * self.dt
        # Evolve velocity according to the force
        self.v[ti + 1] = self.v[ti] + self.accel(ti, Iz) * self.dt

def calc_I(Ii, ti, atoms, zv, dz):
    # Get all atoms' location and velocities
    all_atoms = np.array([[a.n,a.z[ti],a.v[ti]] for a in atoms])
    n_atoms = all_atoms[:,0]
    z_atoms = all_atoms[:,1]
    v_atoms = all_atoms[:,2]
    # Setup for Iz
    Iz = np.ones_like(zv) * Ii
    # calculate intensity at each z step
    for i in range(zv.size-1):
        use_atoms = np.argwhere(np.logical_and(z_atoms>=zv[i],z_atoms<zv[i+1]))[:,0]
        dI = 0
        for j in use_atoms:
            dI += Iz[i] * cst.sigma * n_atoms[j] * dz / (1 + ((2*v_atoms[j]*cst.k) / (cst.gamma))**2 + Iz[i])
        Iz[i+1] = Iz[i] - dI
    return Iz

def calc_n(n, atoms):
    zv = atoms[0].zv
    dz = zv[1]-zv[0]
    for a in atoms:
        for ti in range(a.z.size):
            loc = int(np.round(a.z[ti] / dz))
            if loc < zv.size:
                n[ti,loc] += a.n

def calc_sim_results(I, n0, z, t, n_col):
    # Initial column density
    n_true = n_col
    # Create vectors
    od = np.zeros_like(t)
    If = np.zeros_like(t)
    n_od = np.zeros_like(t)
    n_tot = np.zeros_like(t)
    # Fill in vectors at each time
    for i in range(1,t.size):
        Ii_, If_ = np.sum(I[0:i,0]), np.sum(I[0:i,-1])
        od[i] = np.log(Ii_/If_)
        if not np.isfinite(od[i]): od[i] = 0
        If[i] = 1-If_/Ii_
        n_od[i] = od[i] / cst.sigma / n_true
        n_tot[i] = (od[i] + (Ii_-If_)/(i)) / cst.sigma / n_true
    # Take care of the 0th term
    od[0] = od[1]
    If[0] = If[1]
    n_od[0] = n_od[1]
    n_tot[0] = n_tot[1]
    return (od, If, n_od, n_tot)






