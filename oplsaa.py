# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import re

class atom_attribute(object):
    def __init__(self,names,atomic_type,sigma,epsilon,charge,element,masses):
        self.names = names
        self.atomic_type = atomic_type
        self.epsilon = epsilon
        self.sigma = sigma
        self.charge = charge
        self.element = element
        self.masses = masses
        self.velocity = np.zeros(3)
class pair_interaction(object):
    def __init__(self,atom1,atom2,indices,fudge_factor,Coulombs_factor=None):
        self.sigma = np.sqrt(atom1.sigma*atom2.sigma)
        self.epsilon = np.sqrt(atom1.epsilon*atom2.epsilon)
        self.q_square = atom1.charge * atom2.charge
        self.indices = indices
        self.fudge_factor = fudge_factor
        if Coulombs_factor==None:
            self.Coulombs_factor = 1/(4*np.pi*8.8541878128*1e-12*1e-9)/1000*\
                (1.602176634*1e-19)**2*6.02214076*1e23
        else:
            self.Coulombs_factor = Coulombs_factor
    def compute_E(self,positions = None):
        x1,x2 = positions
        distance_square = sum(np.square(x1 - x2)) 
        distance = np.sqrt(distance_square)
        distance_pow_6 = math.pow(distance_square,3)
        distance_pow_12 = math.pow(distance_pow_6,2)
        sigma_6 = math.pow(self.sigma,6)
        Coulombs = self.Coulombs_factor * self.q_square / (distance) 
        return ((4 * self.epsilon * (math.pow(sigma_6,2)/distance_pow_12\
             -sigma_6/distance_pow_6))*self.fudge_factor['fudgeLJ'],\
                Coulombs * self.fudge_factor['fudgeQQ'])
    def compute_gradient(self,positions = None):
        x1,x2 = positions
        distance_square = sum(np.square(x1 - x2)) 
        distance = np.sqrt(distance_square)
        differ_distance_x = math.pow(distance_square,-0.5)*(x1-x2)
        sigma_6 = math.pow(self.sigma,6)
        distance_pow_6 = math.pow(distance_square,3)
        distance_pow_7 = distance_pow_6 * distance
        distance_pow_12 = math.pow(distance_pow_6,2)  
        distance_pow_13 = distance_pow_12 * distance
        return  (4 * self.epsilon * (-12* math.pow(sigma_6,2)\
               /distance_pow_13+6.0*sigma_6/distance_pow_7) \
               *self.fudge_factor['fudgeLJ']*differ_distance_x,\
                   -1*self.Coulombs_factor * self.fudge_factor['fudgeQQ']\
             * self.q_square / (distance_square) *differ_distance_x )
class bond_terms(object):
    def __init__(self,k,length0,indices):
        self.k = k
        self.length0 = length0
        self.indices = indices
    def compute_E(self,positions = None):
        x1,x2 = positions
        length = np.sqrt(sum(np.square(x1 - x2)))
        return 0.5 * self.k * np.square(self.length0 - length)
    def compute_gradient(self,positions = None):
        x1,x2 = positions
        length_square = sum(np.square(x1 - x2))
        differ_length_x = math.pow(length_square,-0.5)*(x1-x2)
        return self.k*(np.sqrt(length_square)-self.length0) * differ_length_x        
class angle_terms(object):
    def __init__(self,k,angle0,indices):
        self.k = k
        self.angle0 = angle0/180.0*np.pi
        self.indices = indices
    def compute_E(self,positions = None):
        x1,x2,x3 = positions
        r_ij = np.sqrt(sum(np.square(x1 - x2)))
        r_ik = np.sqrt(sum(np.square(x3 - x2)))
        r_jk = np.sqrt(sum(np.square(x3 - x1)))
        costheta = 0.5/(r_ij*r_ik) * (r_ij*r_ij +r_ik*r_ik-r_jk*r_jk)
        if costheta<-1:
            costheta=-1
        elif costheta>1:
            costheta=1      
        return 0.5 * self.k * np.square(self.angle0 - math.acos(costheta))
    def compute_gradient(self,positions = None):
        x1,x2,x3 = positions
        r_ij = np.sqrt(sum(np.square(x1 - x2)))
        r_ik = np.sqrt(sum(np.square(x3 - x2)))
        r_jk = np.sqrt(sum(np.square(x3 - x1)))
        costheta = (0.5/(r_ij*r_ik) * (r_ij*r_ij +r_ik*r_ik-r_jk*r_jk))
        if costheta<-1:
            costheta=-1
        elif costheta>1:
            costheta=1          
        differ_angle_x = [self.differ_angle_edge_x(x1,x2,x3),\
                  self.differ_angle_center_x(x1,x2,x3),\
                  self.differ_angle_edge_x(x3,x2,x1)]
        return [i * self.k * (math.acos(costheta)-self.angle0) * \
                -1/(np.sqrt(1-np.square(costheta))) for i in differ_angle_x]
    @staticmethod
    def differ_angle_edge_x(x1,x2,x3):
        tmp_aa = sum(np.square(x1 - x2))
        tmp_bb = sum(np.square(x2 - x3))
        tmp_cc = sum(np.square(x1 - x3))
        tmp_a = (tmp_aa)**0.5*((tmp_aa) - (tmp_cc) + (tmp_bb))
        tmp_b = (tmp_aa)**1.5
        tmp_c = (tmp_aa)**2.0*(tmp_bb)**0.5
        differ_angle_x = np.array((
        (-0.5*(x1[0] - x2[0])*tmp_a + 1.0*(-x2[0] + x3[0])*tmp_b)/(tmp_c),\
        (-0.5*(x1[1] - x2[1])*tmp_a + 1.0*(-x2[1] + x3[1])*tmp_b)/(tmp_c),\
        (-0.5*(x1[2] - x2[2])*tmp_a + 1.0*(-x2[2] + x3[2])*tmp_b)/(tmp_c)))
        return differ_angle_x
    @staticmethod
    def differ_angle_center_x(x1,x2,x3):
        tmp_aa = (sum(np.square(x1 - x2)))
        tmp_bb = (sum(np.square(x2 - x3)))
        tmp_cc = (sum(np.square(x1 - x3)))
        tmp_a = tmp_aa*tmp_bb**2.0*(tmp_aa - tmp_cc + tmp_bb)
        tmp_b = tmp_aa**2.0*tmp_bb*(tmp_aa - tmp_cc + tmp_bb)
        tmp_c = tmp_aa**2.0*tmp_bb**2.0
        tmp_d = (tmp_aa**2.5*tmp_bb**2.5)
        differ_angle_x = np.array((         
            (0.5*(x1[0] - x2[0])*tmp_a - 0.5*(x2[0] - x3[0])*tmp_b + \
             (-1.0*x1[0] + 2.0*x2[0] - 1.0*x3[0])*tmp_c)/tmp_d, \
            (0.5*(x1[1] - x2[1])*tmp_a - 0.5*(x2[1] - x3[1])*tmp_b + \
             (-1.0*x1[1] + 2.0*x2[1] - 1.0*x3[1])*tmp_c)/tmp_d, \
            (0.5*(x1[2] - x2[2])*tmp_a - 0.5*(x2[2] - x3[2])*tmp_b + \
             (-1.0*x1[2] + 2.0*x2[2] - 1.0*x3[2])*tmp_c)/tmp_d))
        return differ_angle_x
class dihedral_terms(object):
    def __init__(self,c0,c1,c2,c3,c4,c5,indices):
        self.coefficients = [c0,c1,c2,c3,c4,c5]
        self.indices = indices
        self.phi = None
    def compute_E(self,positions = None):
        x1,x2,x3,x4 = positions
        f, g, h = x1-x2, x2-x3, x4-x3
        vec_a = np.cross(f, g)
        vec_b = np.cross(h, g)
        axb = np.cross(vec_a, vec_b)
        cos_tmp = np.dot(vec_a, vec_b)
        sin_tmp = np.dot(axb, g) /  np.linalg.norm(g)
        phi = -np.arctan2(sin_tmp, cos_tmp) 
        dihedral_energy = 0.0
        cos_phi = np.cos(phi+np.pi)
        self.phi = phi
        for i in range(6):
            dihedral_energy += self.coefficients[i] * math.pow(cos_phi,i)
        return dihedral_energy
    def compute_gradient(self,positions = None,):
        if self.phi:
            pass
        else:
            self.compute_E(positions)
        differ_dihedral_x = [self.differ_dihedral_edge_x(positions),\
                    self.differ_dihedral_center_x(positions),\
                    self.differ_dihedral_center_x(positions[::-1]),\
                    self.differ_dihedral_edge_x(positions[::-1])]
        differ_dihedral_x = np.array(differ_dihedral_x).reshape(4,3)
        return self.differ_dihedral_phi(self.coefficients,self.phi) \
                                    * differ_dihedral_x
    @staticmethod
    def differ_dihedral_edge_x(positions):
        (x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4) = positions
        tmp1 = ((x2 - x3)*(y3 - y4) - (x3 - x4)*(y2 - y3))
        tmp2 = ((x2 - x3)*(z3 - z4) - (x3 - x4)*(z2 - z3))
        tmp3 = ((x1 - x2)*(y2 - y3) - (x2 - x3)*(y1 - y2))
        tmp4 = ((x2 - x3)*(-z3 + z4) - (-x3 + x4)*(z2 - z3))
        tmp5 = (-(x1 - x2)*(z2 - z3) + (x2 - x3)*(z1 - z2))
        tmp6 = (-(x2 - x3)*(-y3 + y4) + (-x3 + x4)*(y2 - y3))
        tmp7 = (-(y2 - y3)*(-z3 + z4) + (-y3 + y4)*(z2 - z3))
        tmp8 = ((x2 - x3)*(-y3 + y4) - (-x3 + x4)*(y2 - y3))
        tmp9 = ((y1 - y2)*(z2 - z3) - (y2 - y3)*(z1 - z2))
        tmp10 = (-(x2 - x3)*(z3 - z4) + (x3 - x4)*(z2 - z3))
        tmp11 = ((y2 - y3)*(z3 - z4) - (y3 - y4)*(z2 - z3))
        tmp12 = (-(y2 - y3)*(z3 - z4) + (y3 - y4)*(z2 - z3))
        tmp13 = ((y2 - y3)*(-z3 + z4) - (-y3 + y4)*(z2 - z3))
        tmp14 = ((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
        tmp15 = ((x1 - x2)*(z2 - z3) - (x2 - x3)*(z1 - z2))
        tmp16 = (-(x2 - x3)*(y3 - y4) + (x3 - x4)*(y2 - y3))
        return [-(-((y2 - y3)*tmp1 + (z2 - z3)*tmp2)*((x2 - x3)*\
            (-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 - tmp6*tmp9)\
            + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))/tmp14**0.5 + ((x2 - x3)\
            *(-(y2 - y3)*tmp10 + (z2 - z3)*tmp8) + (y2 - y3)**2*tmp7 + \
            (z2 - z3)**2*tmp7)*(tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)/\
            tmp14**0.5)/((((x2 - x3)*(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)\
            *(tmp3*tmp7 - tmp6*tmp9) + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9)\
            )**2/((tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)**2*tmp14) + 1)*\
            (tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)**2)], [-(-(-(x2 - x3)\
            *tmp1 + (z2 - z3)*tmp11)*((x2 - x3)*(-tmp3*tmp4 + tmp5*tmp6)\
            + (y2 - y3)*(tmp3*tmp7 - tmp6*tmp9) + (z2 - z3)*(-tmp5*tmp7\
            + tmp4*tmp9))/tmp14**0.5 + ((x2 - x3)**2*tmp10 + (y2 - y3)\
            *((x2 - x3)*tmp12 - (z2 - z3)*tmp1) + (z2 - z3)**2*tmp10)*\
            (tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)/tmp14**0.5)/((((x2 - x3)\
            *(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 - tmp6*tmp9)\
            + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))**2/((tmp3*tmp6 + \
            tmp5*tmp4 + tmp9*tmp7)**2*tmp14) + 1)*(tmp3*tmp6 + tmp5*tmp4\
            + tmp9*tmp7)**2)], [-(-(-(x2 - x3)*tmp2 - (y2 - y3)*tmp11)*\
            ((x2 - x3)*(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 -\
            tmp6*tmp9) + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))/tmp14**0.5 \
            + ((x2 - x3)**2*tmp6 + (y2 - y3)**2*tmp1 + (z2 - z3)*((x2 - x3)\
            *tmp13 - (y2 - y3)*tmp10))*(tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)\
            /tmp14**0.5)/((((x2 - x3)*(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)\
            *(tmp3*tmp7 - tmp6*tmp9) + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))\
            **2/((tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)**2*tmp14) + 1)*\
            (tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)**2)]
    @staticmethod
    def differ_dihedral_center_x(positions):
        (x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4) = positions
        tmp1 = ((x2 - x3)*(y3 - y4) - (x3 - x4)*(y2 - y3))
        tmp2 = ((x2 - x3)*(z3 - z4) - (x3 - x4)*(z2 - z3))
        tmp3 = ((x1 - x2)*(y2 - y3) - (x2 - x3)*(y1 - y2))
        tmp4 = ((x2 - x3)*(-z3 + z4) - (-x3 + x4)*(z2 - z3))
        tmp5 = (-(x1 - x2)*(z2 - z3) + (x2 - x3)*(z1 - z2))
        tmp6 = (-(x2 - x3)*(-y3 + y4) + (-x3 + x4)*(y2 - y3))
        tmp7 = (-(y2 - y3)*(-z3 + z4) + (-y3 + y4)*(z2 - z3))
        tmp8 = ((x2 - x3)*(-y3 + y4) - (-x3 + x4)*(y2 - y3))
        tmp9 = ((y1 - y2)*(z2 - z3) - (y2 - y3)*(z1 - z2))
        tmp10 = (-(x2 - x3)*(z3 - z4) + (x3 - x4)*(z2 - z3))
        tmp11 = ((y2 - y3)*(z3 - z4) - (y3 - y4)*(z2 - z3))
        tmp12 = (-(y2 - y3)*(z3 - z4) + (y3 - y4)*(z2 - z3))
        tmp13 = ((y2 - y3)*(-z3 + z4) - (-y3 + y4)*(z2 - z3))
        tmp14 = ((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
        tmp15 = ((x1 - x2)*(z2 - z3) - (x2 - x3)*(z1 - z2))
        tmp16 = (-(x2 - x3)*(y3 - y4) + (x3 - x4)*(y2 - y3))    
        return [-((1.0*(x2 - x3)*((x2 - x3)*(tmp3*tmp10 - tmp15*tmp8) + \
            (y2 - y3)*(tmp3*tmp13 - tmp8*tmp9) + (z2 - z3)*\
            (tmp15*tmp13 - tmp10*tmp9))*tmp14**0.5 + tmp14**1.5*(-(x2 - x3)\
            *((y1 - y3)*tmp2 + (y3 - y4)*tmp15 + (z1 - z3)*tmp16 - \
            (z3 - z4)*tmp3) - (y2 - y3)*((y1 - y3)*tmp11 + (y3 - y4)*tmp9)\
            - (z2 - z3)*((z1 - z3)*tmp11 + (z3 - z4)*tmp9) + tmp3*tmp2 - \
            tmp15*tmp1))*(tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)/tmp14**2.0 - \
            ((x2 - x3)*(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 -\
            tmp6*tmp9) + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))*(-(y1 - y3)*tmp1\
            + (y3 - y4)*tmp3 - (z1 - z3)*tmp2 + (z3 - z4)*tmp15)/tmp14**0.5)\
            /((((x2 - x3)*(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 \
            - tmp6*tmp9) + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))**2/((tmp3*tmp6\
            + tmp5*tmp4 + tmp9*tmp7)**2*tmp14) + 1)*(tmp3*tmp6 + tmp5*tmp4 +\
            tmp9*tmp7)**2)], [-((1.0*(y2 - y3)*((x2 - x3)*(tmp3*tmp10 -\
            tmp15*tmp8) + (y2 - y3)*(tmp3*tmp13 - tmp8*tmp9) + (z2 - z3)*\
            (tmp15*tmp13 - tmp10*tmp9))*tmp14**0.5 + tmp14**1.5*(-(x2 - x3)\
            *((x1 - x3)*tmp10 - (x3 - x4)*tmp15) - (y2 - y3)*((x1 - x3)*tmp13\
            - (x3 - x4)*tmp9 + (z1 - z3)*tmp16 - (z3 - z4)*tmp3) - (z2 - z3)*\
            ((z1 - z3)*tmp10 - (z3 - z4)*tmp15) - tmp3*tmp12 + tmp16*tmp9))*\
            (tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)/tmp14**2.0 - ((x2 - x3)*\
            (-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 - tmp6*tmp9) + \
            (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))*((x1 - x3)*tmp1 - (x3 - x4)*\
            tmp3 - (z1 - z3)*tmp11 + (z3 - z4)*tmp9)/tmp14**0.5)/((((x2 - x3)\
            *(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 - tmp6*tmp9) + \
            (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))**2/((tmp3*tmp6 + tmp5*tmp4 + \
            tmp9*tmp7)**2*tmp14) + 1)*(tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)**2)]\
            , [-((1.0*(z2 - z3)*((x2 - x3)*(tmp3*tmp10 - tmp15*tmp8) + \
            (y2 - y3)*(tmp3*tmp13 - tmp8*tmp9) + (z2 - z3)*(tmp15*tmp13 -\
            tmp10*tmp9))*tmp14**0.5 + tmp14**1.5*((x2 - x3)*((x1 - x3)*tmp16\
            - (x3 - x4)*tmp3) - (y2 - y3)*((y1 - y3)*tmp1 + (y3 - y4)*tmp3) \
            - (z2 - z3)*((x1 - x3)*tmp13 - (x3 - x4)*tmp9 - (y1 - y3)*tmp10 \
            + (y3 - y4)*tmp15) - tmp15*tmp12 - tmp2*tmp9))*(tmp3*tmp6 +\
            tmp5*tmp4 + tmp9*tmp7)/tmp14**2.0 - ((x2 - x3)*(-tmp3*tmp4 +\
            tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 - tmp6*tmp9) + (z2 - z3)*\
            (-tmp5*tmp7 + tmp4*tmp9))*((x1 - x3)*tmp2 - (x3 - x4)*tmp15 \
            + (y1 - y3)*tmp11 - (y3 - y4)*tmp9)/tmp14**0.5)/((((x2 - x3)\
            *(-tmp3*tmp4 + tmp5*tmp6) + (y2 - y3)*(tmp3*tmp7 - tmp6*tmp9)\
            + (z2 - z3)*(-tmp5*tmp7 + tmp4*tmp9))**2/((tmp3*tmp6 + tmp5*tmp4\
            + tmp9*tmp7)**2*tmp14) + 1)*(tmp3*tmp6 + tmp5*tmp4 + tmp9*tmp7)**2)]
    @staticmethod
    def differ_dihedral_phi(coefficients,phi):     
        c0,c1,c2,c3,c4,c5 = coefficients 
        return (c1 - 2*c2*np.cos(phi) + 3*c3*math.pow(np.cos(phi),2) - \
                            4*c4*math.pow(np.cos(phi),3))*np.sin(phi)
class improper_terms(object):
    def __init__(self):
        pass       
class Forcefield(object):
    def __init__(self,ff_names = "opls-aa"):
        self.atom_types = {"CT":"C","HC":"H"}
        self.atom_names = ["opls_136","opls_135","opls_140"]
        self.atom_masses = {"CT":12.01,"HC":1.008}
        #mixing rule for C-H, mix geometric
        self.vdw_parameter = {"opls_136":[0.35,0.2761440],\
                  "opls_135":[0.35,0.2761440],"opls_140":[0.25,0.1255200]} 
            #sigma (nm) ,epsilon (kJ/mol)
        self.charge_parameter = {"opls_136":-0.1200,\
                  "opls_135":-0.1800,"opls_140":0.0600} 
        self.cut_off = 1.1 #(nm)
        self.bond_params = {"CT-CT":[224262.4,0.1529],"CT-HC":[284512.0,0.109]} 
        self.angle_params = {"CT-CT-CT":[112.700,488.273],\
                             "CT-CT-HC":[110.700,313.800],\
                             "HC-CT-HC":[107.800,276.144]}
        self.dihedral_params={"CT-CT-CT-CT":[2.9288,-1.4644,0.2092,-1.6736,0,0]\
                             ,"CT-CT-CT-HC":[0.6276,1.8828,0,-2.5104,0,0],
                             "HC-CT-CT-HC":[0.6276,1.8828,0,-2.5104,0,0]}
        self.fudge_factor = {"fudgeLJ":0.5, "fudgeQQ":0.5}
        self.Coulombs_factor = 1/(4*np.pi*8.8541878128*1e-12*1e-9)/1000*\
                                    (1.602176634*1e-19)**2*6.02214076*1e23
        self.bond_criterion = {}
        for key,value in self.bond_params.items(): 
            tmp = key.split("-")
            self.bond_criterion["%s-%s" %(self.atom_types[tmp[0]],\
                  self.atom_types[tmp[1]])] = [0.9*value[1],1.1*value[1]]
        tobe_append = {}
        for key,value in self.bond_criterion.items():
            tmp = key.split("-")
            if "%s-%s" %(tmp[1],tmp[0]) != key:
                tobe_append.update({"%s-%s" %(tmp[1],tmp[0]):value})
        self.bond_criterion.update(tobe_append) 
        tobe_append = {}
        for key,value in self.bond_params.items():
            tmp = key.split("-")
            if "%s-%s" %(tmp[1],tmp[0]) != key:
                tobe_append.update({"%s-%s" %(tmp[1],tmp[0]):value})
        self.bond_params.update(tobe_append)     
        tobe_append = {}
        for key,value in self.angle_params.items():
            tmp = key.split("-")
            if "%s-%s-%s" %(tmp[2],tmp[1],tmp[0]) != key:
                tobe_append.update({"%s-%s-%s" %(tmp[2],tmp[1],tmp[0]):value})
        self.angle_params.update(tobe_append)  
        tobe_append = {}
        for key,value in self.dihedral_params.items():
            tmp = key.split("-")
            if "%s-%s-%s-%s" %(tmp[3],tmp[2],tmp[1],tmp[0]) != key:
                tobe_append.update({"%s-%s-%s-%s" %(tmp[3],tmp[2],tmp[1]\
                                                    ,tmp[0]):value})
        self.dihedral_params.update(tobe_append)          
    def parameterize(self,structure):
        distance_map = []
        for index_i in range(len(structure.xyz_sets)):
            for index_j in range(index_i+1,len(structure.xyz_sets)):
                distance = np.sqrt(sum(np.square(structure.xyz_sets[index_i] -\
                                        structure.xyz_sets[index_j])))
                distance_map.append((index_i,index_j,distance))
        structure.distance_map = []
        for i,j,d in distance_map:
            seq = "%s-%s" %(structure.atom_labels[i],structure.atom_labels[j])
            if seq not in self.bond_criterion.keys():
                continue
            if d < max(self.bond_criterion[seq]) and \
                                            d > min(self.bond_criterion[seq]):
                structure.distance_map.append((i,j,d))
        #atom typing        
        for index,i in enumerate(structure.atom_labels):
            if i == "H":
                this_atom = atom_attribute("opls_140","HC",\
                         *self.vdw_parameter["opls_140"],self.charge_parameter\
                      ["opls_140"],i,self.atom_masses["HC"])
            elif i == "C":
                tmp =[set(j[:2])-set([index]) for j in structure.distance_map]
                number_of_bonded_hygrogen = [structure.atom_labels[list(i)[0]]\
                                         for i in tmp if len(i)==1].count("H")
                if number_of_bonded_hygrogen==3:
                    this_atom = atom_attribute("opls_135","CT",\
                         *self.vdw_parameter["opls_135"],self.charge_parameter\
                          ["opls_135"],i,self.atom_masses["CT"])
                else:
                    this_atom = atom_attribute("opls_136","CT",\
                         *self.vdw_parameter["opls_136"],self.charge_parameter\
                          ["opls_136"],i,self.atom_masses["CT"])                        
            structure.atom_sets.append(this_atom)  
        #bond typing 
        for i,j,_ in structure.distance_map:
                seq = "%s-%s" %(structure.atom_sets[i].atomic_type,\
                                 structure.atom_sets[j].atomic_type)
                this_bond = bond_terms(*self.bond_params[seq],(i,j))
                structure.bond_sets.append(this_bond)
        #angle typing 
        angle_pairs= self.search_angles(structure)
        #dihedral typing 
        dihedral_pairs = self.search_dihedral(angle_pairs,structure)
        tmp_1 = []
        for ii,_,_,jj in dihedral_pairs:
            tmp_1.append(sorted([ii,jj]))
        structure.pairs_1_4 = tmp_1
        connect_pairs = []
        for ii_ in structure.distance_map:
            connect_pairs.append(list(ii_[:2]))
        for i in angle_pairs:
            connect_pairs.append([i[0],i[2]])
        connect_pairs+=structure.pairs_1_4
        for index_i in range(len(structure.atom_labels)):
            for index_j in range(index_i+1,len(structure.atom_labels)):
                if [index_i,index_j] in connect_pairs or [index_j,index_i] \
                    in connect_pairs:
                        pass
                else:
                    structure.non_1_4_pairs.append((index_i,index_j))
        structure.non_1_4_pairs = list(set(structure.non_1_4_pairs))
        #prevent ring
        structure.pairs_1_4 =list(set([tuple(i) for i in structure.pairs_1_4]))
        for i,j in structure.pairs_1_4:
            this_pair = pair_interaction(structure.atom_sets[i],\
                            structure.atom_sets[j],(i,j),self.fudge_factor,\
                                self.Coulombs_factor)
            structure.pairs_1_4_sets.append(this_pair)
        for i,j in structure.non_1_4_pairs:
            this_pair = pair_interaction(structure.atom_sets[i],\
                            structure.atom_sets[j],(i,j),{"fudgeLJ":1.0, \
                                "fudgeQQ":1.0}, self.Coulombs_factor)
            structure.non_1_4_pairs_sets.append(this_pair)
    def search_angles(self,structure):
        # serach center atom *-i-*  
        angle_pairs = []
        for index,i in enumerate(structure.atom_sets):
            tmp =[set(j[:2])-set([index]) for j in structure.distance_map]
            neighbors = [ii for ii in tmp if len(ii)==1]
            if len(neighbors) ==1:
                continue
            else:
                for ii in range(len(neighbors)):
                    for jj in range(ii+1,len(neighbors)):
                        left = list(neighbors[ii])[0]
                        right = list(neighbors[jj])[0]
                        seq="%s-%s-%s"%(structure.atom_sets[left].atomic_type,\
                                       structure.atom_sets[index].atomic_type,\
                                       structure.atom_sets[right].atomic_type)
                        angle_pairs.append((left,index,right))
                        this_angle = angle_terms(*list(reversed(\
                                  self.angle_params[seq])),(left,index,right))
                        structure.angle_sets.append(this_angle)
        return angle_pairs
    def search_dihedral(self,angle_pairs,structure):
        # serach center atom *-i-* 
        dihedral_pairs = []
        for i,j,k in angle_pairs:
            maybe_paired_atoms_left = [list(ii)[0] for ii in [set(jj[:2])\
                     -set([i]) for jj in structure.distance_map] if len(ii)==1]
            maybe_paired_atoms_right = [list(ii)[0] for ii in [set(jj[:2])\
                     -set([k]) for jj in structure.distance_map] if len(ii)==1] 
            maybe_paired_atoms_left = set(maybe_paired_atoms_left)-set([j])  
            maybe_paired_atoms_right = set(maybe_paired_atoms_right)-set([j])
            if maybe_paired_atoms_left != set():
                for ii in maybe_paired_atoms_left:
                    if ii<k:
                        dihedral_pairs.append([ii,i,j,k])
                    else:
                        dihedral_pairs.append(list(reversed([ii,i,j,k])))
            if maybe_paired_atoms_right != set():
                for ii in maybe_paired_atoms_right:
                    if i<ii:
                        dihedral_pairs.append([i,j,k,ii])
                    else:
                        dihedral_pairs.append(list(reversed([i,j,k,ii])))                    
        dihedral_pairs = (list(set([tuple(i) for i in dihedral_pairs])))
        for i,j,k,m in dihedral_pairs:
                seq = "%s-%s-%s-%s" %(structure.atom_sets[i].atomic_type,\
                                 structure.atom_sets[j].atomic_type,\
                                 structure.atom_sets[k].atomic_type,\
                                 structure.atom_sets[m].atomic_type)
                this_dihedral = dihedral_terms(*self.dihedral_params[seq],\
                                               (i,j,k,m))
                structure.dihedrals_sets.append(this_dihedral)
        return dihedral_pairs                 
class Structure(object):
    def __init__(self):
        self.atom_nums = 0
        self.boundary = None
        self.atom_sets = []
        self.atom_labels = []
        self.xyz_sets = []
        self.bond_sets = []
        self.angle_sets = []
        self.dihedrals_sets = []
        self.pairs_1_4_sets = []
        self.pairs_1_4 = None
        self.non_1_4_pairs = []
        self.non_1_4_pairs_sets = []
        self.distance_map = None   
    def read_structure(self,file,fmt="xyz"):
        if fmt != "xyz":
            raise SystemError("Error! Only xyz file is supported~")
        if not os.path.exists(file):
            raise IOError("Error! File not exists.")
        with open(file,'r') as reader:
            self.atom_nums = int(reader.readline())
            tmp = reader.readline()
            found_lattice = False
            if "Lattice" in tmp:
                found_lattice = True 
            elif "LATTICE" in tmp:
                found_lattice = True  
            elif "lattice" in tmp:
                found_lattice = True  
            re_results = []
            if found_lattice:    
                re_pattern = re.compile("Lattice.*?=[\"\'](.*?)[\"\']")
                re_results = re.findall(re_pattern, tmp)
            if re_results == []:
                found_lattice = False
            else:
                tmp = []
                for ii in re_results[0].split():
                    tmp.append(float(ii))
                self.boundary = np.array(tmp).reshape(-1,3)
            for i in range(self.atom_nums):
                tmp = reader.readline().split()
                self.atom_labels.append(tmp[0])
                self.xyz_sets.append(list(map(float,tmp[1:4])))
            self.xyz_sets = np.array(self.xyz_sets)/10.0
class Calc_energy_and_forces(object):
    def __init__(self,structure,forcefield):
        self.pair_energy = 0.0
        self.bond_energy = 0.0
        self.angle_energy = 0.0
        self.dihedral_energy = 0.0
        self.structure = structure 
        self.forcefield = forcefield
        self.atomic_forces = np.zeros(3)
    def calc_energy(self,xyz_sets):
        total_atoms = len(xyz_sets)
        for index,i in enumerate(self.structure.pairs_1_4_sets):
            self.pair_energy += sum(i.compute_E((xyz_sets[i.indices[0]],\
                                            xyz_sets[i.indices[1]])))                                          
        for index,i in enumerate(self.structure.non_1_4_pairs_sets):
            self.pair_energy += sum(i.compute_E((xyz_sets[i.indices[0]],\
                                            xyz_sets[i.indices[1]])))
        for index,i in enumerate(self.structure.bond_sets):
            self.bond_energy += i.compute_E((xyz_sets[i.indices[0]],\
                                            xyz_sets[i.indices[1]]))                                           
        for index,i in enumerate(self.structure.angle_sets):
            self.angle_energy += i.compute_E((xyz_sets[i.indices[0]],\
                            xyz_sets[i.indices[1]],xyz_sets[i.indices[2]]))                      
        for index,i in enumerate(self.structure.dihedrals_sets):
            self.dihedral_energy += i.compute_E((xyz_sets[i.indices[0]],\
                     xyz_sets[i.indices[1]],xyz_sets[i.indices[2]],\
                            xyz_sets[i.indices[3]]))  
        total_energy = self.pair_energy+self.bond_energy+self.angle_energy+\
                                self.dihedral_energy                               
        return total_energy
    def calc_atomic_forces(self,xyz_sets):  
        total_atoms = len(xyz_sets)
        atomic_forces = dict(zip(range(total_atoms),np.zeros((total_atoms,3))))
        for index,i in enumerate(self.structure.pairs_1_4_sets):
            ret = ((i.compute_gradient((xyz_sets[i.indices[0]],\
                                            xyz_sets[i.indices[1]]))))
            atomic_forces[i.indices[0]] += -1*(ret[0]+ret[1])
            atomic_forces[i.indices[1]] += (ret[0]+ret[1])
        for index,i in enumerate(self.structure.non_1_4_pairs_sets):
            ret = i.compute_gradient((xyz_sets[i.indices[0]],\
                                            xyz_sets[i.indices[1]]))
            atomic_forces[i.indices[0]] += -1*(ret[0]+ret[1])
            atomic_forces[i.indices[1]] += (ret[0]+ret[1])   

        for index,i in enumerate(self.structure.bond_sets):
            ret = i.compute_gradient((xyz_sets[i.indices[0]],\
                                            xyz_sets[i.indices[1]]))
            atomic_forces[i.indices[0]] += -1*ret
            atomic_forces[i.indices[1]] += ret           
        for index,i in enumerate(self.structure.angle_sets):
            ret = i.compute_gradient((xyz_sets[i.indices[0]],\
                    xyz_sets[i.indices[1]],xyz_sets[i.indices[2]]))             
            atomic_forces[i.indices[0]] += -1*ret[0]
            atomic_forces[i.indices[1]] += -1*ret[1]
            atomic_forces[i.indices[2]] += -1*ret[2]
        for index,i in enumerate(self.structure.dihedrals_sets):
            ret = i.compute_gradient((xyz_sets[i.indices[0]],\
                    xyz_sets[i.indices[1]],xyz_sets[i.indices[2]]\
                    ,xyz_sets[i.indices[3]]))
            atomic_forces[i.indices[0]] += -1*ret[0]
            atomic_forces[i.indices[1]] += -1*ret[1]
            atomic_forces[i.indices[2]] += -1*ret[2]
            atomic_forces[i.indices[3]] += -1*ret[3]
        return np.array([atomic_forces[i] for i in range(total_atoms)])
if __name__ == "__main__":
    ff = Forcefield("opls-aa")                
    structure = Structure()
    structure.read_structure("propane.xyz")
    ff.parameterize(structure)
    calculator = Calc_energy_and_forces(structure,ff)
    total_energy = calculator.calc_energy(structure.xyz_sets)
    atomic_forces = calculator.calc_atomic_forces(structure.xyz_sets)
    total_atoms = len(structure.xyz_sets)
    atomic_forces = np.array([atomic_forces[i] for i in range(total_atoms)])
    print(total_energy,"kJ/mol")
    print(atomic_forces,"kJ/mol/nm")