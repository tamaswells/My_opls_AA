# -*- coding: utf-8 -*-

import os
import numpy as np
from oplsaa import *

class MD_engine(object):
    def __init__(self,structure,calculator):
        # T: temperature in units of K 
        self.structure = structure
        self.calculator = calculator
        self.constants = {"K_B":8.625e-5,'fs_factor':1/10.18,\
            "kj-mol2eV":0.01036427230133138,\
            'force_factor':0.01036427230133138/10}            
    def initialize_velocity(self,T):
        masses = np.array([i.masses for i in self.structure.atom_sets])
        total_atoms = len(masses)
        velocities = np.random.uniform(-1,1,(total_atoms,3))
        xyz = self.structure.xyz_sets*10
        velocities = self.correct_velocity(xyz,masses,velocities)
        #scale velocity according T
        scale_factor = np.dot(masses,np.sum(velocities**2,axis=1))
        velocities = velocities * np.sqrt(T*3.0*self.constants['K_B']\
                            *total_atoms/scale_factor) 
        return velocities #eV^(1/2)amu(−1/2)
    @staticmethod
    def zero_linear_momentum(masses,velocities):
        center_of_mass_velocity=np.zeros(3)
        total_atoms = len(masses)
        for d in range(3):
            center_of_mass_velocity[d] += np.dot(velocities[:,d],masses)
        center_of_mass_velocity /= sum(masses)
        for i in range(total_atoms):
            velocities[i] -= center_of_mass_velocity    
        return velocities 
    @staticmethod
    def get_angular_momentum(c_o_m,xyz,masses,velocities):
        angular_momentum = np.zeros(3)
        total_atoms = len(masses)
        for i in range(total_atoms):
            dx = xyz[i,0] - c_o_m[0]
            dy = xyz[i,1] - c_o_m[1]
            dz = xyz[i,2] - c_o_m[2]
            angular_momentum[0] += masses[i]*(dy*velocities[i,2]\
                    -dz*velocities[i,1]) 
            angular_momentum[1] += masses[i]*(dz*velocities[i,0]\
                    -dx*velocities[i,2])  
            angular_momentum[2] += masses[i]*(dx*velocities[i,1]\
                    -dy*velocities[i,0])     
        return angular_momentum
    @staticmethod
    def get_moment_inertia(c_o_m,xyz,masses):
        inertia = np.zeros((3,3))
        total_atoms = len(masses)
        for i in range(total_atoms):
            dx = xyz[i,0] - c_o_m[0]
            dy = xyz[i,1] - c_o_m[1]
            dz = xyz[i,2] - c_o_m[2]
            inertia[0,0] += masses[i] * (dy * dy + dz * dz)
            inertia[1,1] += masses[i] * (dx * dx + dz * dz)
            inertia[2,2] += masses[i] * (dx * dx + dy * dy)
            inertia[0,1] -= masses[i] * dx * dy
            inertia[1,2] -= masses[i] * dy * dz
            inertia[0,2] -= masses[i] * dx * dz   
        inertia[1,0] = inertia[0,1]
        inertia[2,1] = inertia[1,2]
        inertia[2,0] = inertia[0,2]
        return inertia
    @staticmethod
    def get_angular_velocity(inertia,angular_momentum):
        determinant = np.linalg.det(inertia)
        if (determinant > -1.0e-10 and determinant < 1.0e-10):
            return (False,0) #do not correct the angular velocity to avoid NaN
        else:
            inverse = np.linalg.inv(inertia)
            angular_velocity = np.zeros(3)
            angular_velocity[0] = inverse[0,0] * angular_momentum[0] + inverse[0,1] \
                * angular_momentum[1] + inverse[0,2] * angular_momentum[2]
            angular_velocity[1] = inverse[1,0] * angular_momentum[0] + inverse[1,1] \
                * angular_momentum[1] + inverse[1,2] * angular_momentum[2]
            angular_velocity[2] = inverse[2,0] * angular_momentum[0] + inverse[2,1] \
                * angular_momentum[1] + inverse[2,2] * angular_momentum[2]
            return (True,angular_velocity)
    @staticmethod
    def zero_angular_momentum(angular_velocity,c_o_m,xyz,velocities):
        total_atoms = len(xyz)
        for i in range(total_atoms):
            dx = xyz[i,0] - c_o_m[0]
            dy = xyz[i,1] - c_o_m[1]
            dz = xyz[i,2] - c_o_m[2]
            velocities[i][0] -= angular_velocity[1] *dz - angular_velocity[2]*dy
            velocities[i][1] -= angular_velocity[2] *dx - angular_velocity[0]*dz
            velocities[i][2] -= angular_velocity[0] *dy - angular_velocity[1]*dx
        return velocities
    def correct_velocity(self,xyz,masses,velocities):
        #remove linear momentum
        velocities = self.zero_linear_momentum(masses,velocities)
        #center of mass position
        c_o_m = np.zeros(3)
        for d in range(3):
            c_o_m[d] += np.dot(xyz[:,d],masses)
        c_o_m /= sum(masses) 
        #angular momentum
        angular_momentum = self.get_angular_momentum(c_o_m,xyz,\
                                        masses,velocities)
        # moment of inertia
        inertia = self.get_moment_inertia(c_o_m,xyz,masses)
        do_angle_correct,angular_velocity = self.get_angular_velocity\
                    (inertia,angular_momentum)
        if do_angle_correct:
            velocities = self.zero_angular_momentum(angular_velocity,c_o_m,\
                                xyz,velocities)
        return velocities
        
    def integrate(self,Ne,Np,Ns,dt,T):
        #calculate initial energy and forces 
        potential_energy=self.calculator.calc_energy(self.structure\
                      .xyz_sets)*self.constants['kj-mol2eV']
        forces = self.calculator.calc_atomic_forces(self.structure.xyz_sets)\
                            *self.constants['force_factor']
        velocities = self.initialize_velocity(T)
        masses = np.array([i.masses for i in self.structure.atom_sets])
        total_atoms = len(masses)
        dt = dt*self.constants['fs_factor']
        print("Now run MD simulations...")
        E = np.zeros((int(Np/Ns),3)) # energy data to be computed
        if os.path.exists("trajectory.xyz"):
            os.remove("trajectory.xyz")
        for step in range(Ne+Np): #time-evolution started
            #step 1 of Velocity-Verlet
            for d in range(3): # step 1 of Velocity-Verlet
                velocities[:,d]+=(forces[:,d]/masses)*(dt*0.5)
                self.structure.xyz_sets[:,d]+=velocities[:,d]*dt/10
            potential_energy=self.calculator.calc_energy(self.structure.\
                                xyz_sets)*self.constants['kj-mol2eV']
            forces = self.calculator.calc_atomic_forces(self.structure.\
                                xyz_sets)*self.constants['force_factor']            
            #update forces
            for d in range(3): # step 2 of Velocity-Verlet
                velocities[:,d]+=(forces[:,d]/masses)*(dt*0.5)
            if step<=Ne:    #control temperature in the equilibration stage
                #scale velocity according T
                scale_factor = np.dot(masses,np.sum(velocities**2,axis=1))
                velocities = velocities * np.sqrt(T*3.0*self.constants\
                                    ['K_B']*total_atoms/scale_factor)                 
            elif np.mod(step,Ns)==0: #measure in the production stage
                E[int((step-Ne)/Ns),0]=potential_energy/total_atoms 
                E[int((step-Ne)/Ns),1]=0.5*np.dot(masses,np.sum(\
                      velocities**2 ,axis=1))/total_atoms # kinetic energy
                self.write_trajectory(time=step*dt/self.constants['fs_factor'])
        # time-evolution completed
        E[:,2]=E[:,1]+E[:,0] # total enegy (per atom)   
        return E
    def write_trajectory(self,time):
        with open("trajectory.xyz",'a+') as writer:
            writer.write("%d\n"%(len(self.structure.atom_labels)))
            writer.write("Time='%f'\n" %round(time,2))
            for index, i in enumerate(self.structure.atom_labels):
                xyz = self.structure.xyz_sets * 10
                writer.write("%s %f %f %f\n" %(i,xyz[index][0],\
                                xyz[index][1],xyz[index][2]))
        
if __name__ == "__main__":
    #setup forcefield
    ff = Forcefield("opls-aa")                
    structure = Structure()
    #read_strcuture
    structure.read_structure("C3/propane.xyz")
    ff.parameterize(structure)
    calculator = Calc_energy_and_forces(structure,ff)
    # T: is temperature in units of K
    # Ne: number of time steps in the equilibration stage
    # Np: number of time steps in the production stage
    # Ns: sampling interval in the production stage
    # dt: time step of integration in units of fs   
    md_engine = MD_engine(structure,calculator)
    E = md_engine.integrate(Ne=1000,Np=1000,Ns=20,dt=1,T = 300)
    #print(E)
