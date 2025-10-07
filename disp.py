import sys

import numpy as np
from scipy.optimize import root
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

import h5py

from interpolate_table import interpolate_eas

from typing import Tuple, Optional

import csv



# Module-scope declarations:

# --- physical constants and conversion ---
GF_GEVm2 = 1.1663787e-5 #GeV^-2
SQRT2_GF_GEVm2 = np.sqrt(2.0) * 1.1663787e-5   # GeV^-2 (root2 * G_F)
GF_MeV_m2 = SQRT2_GF_GEVm2 * 1e-6              # MeV^-2
MEV_TO_CM_INV = 5.067730716e10                 # 1 MeV = 5.0677e10 cm^-1
C_PREFAC = (GF_MeV_m2 / (2.0 * np.pi**2)) * MEV_TO_CM_INV  #(1/2*pi^2) included in this prefactor
HBAR_C_MeV_cm = 1.973269804e-11   # MeV*cm
ERG_TO_MEV    = 6.241509074e5
PI = np.pi
C_f= 1.0/(2.0 * PI**2 * (HBAR_C_MeV_cm**3)) # MeV^-3 cm^-3
c= 2.99792458e10 # speed of light in cm/s


HBAR_MeV_s = 6.582119569e-22         # MeV*s
MEV_M2_TO_CM3_PER_S = (HBAR_C_MeV_cm**3) / HBAR_MeV_s   # 1 MeV^-2 -> cm^3/s
GEV_M2_TO_CM3_PER_S = MEV_M2_TO_CM3_PER_S * 1e6         # 1 GeV^-2 -> cm^3/s

GF_cm3_per_s = GF_MeV_m2 * MEV_M2_TO_CM3_PER_S/np.sqrt(2.0)


#placeholders

E_mid: Optional[np.ndarray] = None
bin_bottom: Optional[np.ndarray] = None
bin_top: Optional[np.ndarray] = None
bin_widths: Optional[np.ndarray] = None
W3: Optional[np.ndarray] = None
continuous= False
trial_model= False
x_cm=0.0
y_cm=0.0
z_cm=0.0
mod_op= False


verbose=0


def load_model_nJ(model_path: str, species: str, idx: Tuple[int,int,int], include_state: bool=False):
   """
   Read number density n [1/ccm] and energy density J [erg/ccm]
   for species in {'e','x','a'} at voxel (ix,iy,iz) from model_rl1_orthonormal.h5.
   Returns: n_cm3, u_MeV_cm3
   """
   key_n = f"n_{species}(1|ccm)"
   key_J = f"J_{species}(erg|ccm)"
   ix, iy, iz = idx

   with h5py.File(model_path, "r") as f:
     n = f[key_n][ix, iy, iz]               # [1/cm^3]
     J_erg = f[key_J][ix, iy, iz]          # [erg/cm^3]

     if include_state:
       rho_cgs = float(f["rho(g|ccm)"][ix, iy, iz])   # [g/cm^3]
       T_MeV   = float(f["T(MeV)"][ix, iy, iz])       # [MeV]
       Ye      = float(f["Ye"][ix, iy, iz])           # [dimensionless]
       n_x= float (f["n_x(1|ccm)"][ix,iy,iz])
       n_a= float (f["n_a(1|ccm)"][ix,iy,iz])
       x_cm_temp= float (f["x(cm)"][ix,iy,iz])
       y_cm_temp= float (f["y(cm)"][ix,iy,iz])
       z_cm_temp= float (f["z(cm)"][ix,iy,iz])

   # convert energy density to MeV/cm^3
   u_mev = J_erg * ERG_TO_MEV                              # [MeV/cm^3]
    



   if include_state:
     return n, n_a, n_x, u_mev, rho_cgs, T_MeV, Ye, x_cm_temp, y_cm_temp, z_cm_temp

   return n, u_mev  
   







def _nearest_idx(arr, x):
   a = np.asarray(arr)
   return int(np.abs(a - x).argmin())



def FD_dist(E, g, T, mu=0.0):
   
   x= E/T - mu
   
   f= g/(np.exp(np.clip(x, -700, 700)) + 1)

   return f

def phi( E, theta):
   ''' f= A*phi. theta=kT. A is similar to gbar'''
   return 1.0 / (np.exp(E / theta) + 1.0)

   
def solve_A_from_n(n_cm3, theta, E_mid, W3):
   """
   n = C_f * A * (1/3) * sum(phi* W3)  ->  A = n * 3 / (C_f * sum( phi* W3))
   """
   if continuous is True:

     return float(n_cm3/(C_f * theta**3 * 1.803085))


   S1 = float(np.sum(phi(E_mid, theta) * W3))
   
   return n_cm3 * 3.0 / (C_f * S1)

  
def solve_A_from_u(u_mev_cm3, theta, E_mid, W3):
   """
   u = C * A * (1/3) * sum( phi* E_mid W3)  ->  A = u * 3 / (C * sum(phi* E_mid W3))
   """
   
   if continuous is True:
     
     return u_mev_cm3/(C_f * theta**4 * 5.682195)     

   S2 = float(np.sum(phi(E_mid, theta) * E_mid * W3))
                            
   return u_mev_cm3 * 3.0 / (C_f * S2)



def solve_theta(u_over_n_mev, E_mid, W3):
   """
   Solve (sum(phi E W3))/(sum( phi W3)) = u/n  for theta (MeV).
   """
   target = float(u_over_n_mev)

   if continuous is True:
     theta= float(1.803085*u_over_n_mev/5.682195)
     return theta, 0.0

   def r(theta):
     phi = 1.0 / (np.exp(E_mid / theta) + 1.0)
     return (np.sum(phi * E_mid * W3) /np.sum(phi * W3))
   
   # bracket guess
   theta0 = max(1e-6, 0.317311 * target)  # ~ (F2/F3)*(u/n), in MeV
   theta_lo = theta0 / 50.0               # cold bound
   theta_hi = theta0 * 50.0               # hot bound
   
   f_lo = r(theta_lo) - target
   f_hi = r(theta_hi) - target
   
   if f_lo * f_hi < 0.0:
     theta= float(brentq(lambda t: r(t) - target, theta_lo, theta_hi))
     return theta,  0

   # Fallback if target is outside achievable range on this grid
   #print("Bracketing failed. Choosing theta closest to target")
   thetas = np.logspace(np.log10(theta_lo), np.log10(theta_hi), 200)
   rvals  = np.array([r(t) for t in thetas])
   return float(thetas[np.argmin(np.abs(rvals - target))]), 1


def spectral_dist_fit(model_path, species, idx):

   if "E_mid" not in globals() or E_mid is None or "W3" not in globals() or W3 is None:
     raise RuntimeError("E_mid/W3 not set. Call load_gamma_from_nulib(...) first.")
   
   n_cm3, u_mev_cm3 = load_model_nJ(model_path, species, idx)

   target = u_mev_cm3 / n_cm3
   theta, bracket_fail = solve_theta(target, E_mid, W3)

   A_n = solve_A_from_n(n_cm3, theta, E_mid, W3)
   A_u = solve_A_from_u(u_mev_cm3, theta, E_mid, W3)

   A = 0.5 * (A_n + A_u)

   f_E = A / (np.exp(E_mid / theta) + 1.0)
        
   return theta, A, E_mid, f_E, bracket_fail



def I_omega(omega, E, del_f, Gamma):
  
   #global bin_widths 

   denom= omega + 1j*Gamma
   integrand = (E**2)*del_f/denom

   if bin_widths is None:
     term = np.trapz(integrand, E)
   else:
     term = np.sum(integrand * bin_widths)
    
   return C_PREFAC * term 
       

def solve_dispersion(E, del_f, Gamma, target, w0=0.0+1j*1e-3):
   """
   Solve I(w) = target (default -1).
   Returns w and a small info dict.
   """
   def F(xy):
     wr, wi = xy
     w = complex(wr, wi)
     Iw = I_omega(w, E, del_f, Gamma)
     return [Iw.real - target, Iw.imag]
    
   sol = root(F, [w0.real, w0.imag], method="hybr")
   w = sol.x[0] + 1j*sol.x[1]
    
   return w, sol

def I_omega_twofluids(omega, E, df_nu, Gam_nu, df_anu, Gam_anu):
   
   #global bin_widths 

   denom_nu  = omega + 1j * Gam_nu
   denom_anu = omega + 1j * Gam_anu
   
   if bin_widths is None:

     term_nu  = np.trapz((E**2) * df_nu  / denom_nu,  E)
     term_anu = np.trapz((E**2) * df_anu / denom_anu, E)

   else:
      #print(f"bin_widths is true")
      term_nu  = np.sum((E**2) * df_nu  / denom_nu * bin_widths)
      term_anu = np.sum((E**2) * df_anu / denom_anu * bin_widths)
   
   return (C_PREFAC * (term_nu - term_anu))


def solve_dispersion_twofluids(E, df_nu, Gam_nu, df_anu, Gam_anu, target, w0=0.0+1e-6j):
                        
   def F(xy):
     wr, wi = xy
     w = complex(wr, wi)
     Iw = I_omega_twofluids(w, E, df_nu, Gam_nu, df_anu, Gam_anu)
     return [Iw.real - target, Iw.imag]

   sol = root(F, [w0.real, w0.imag], method="hybr")
   w = sol.x[0] + 1j*sol.x[1]

   return w, sol


def load_gamma_from_nulib(nulib_path, rho_cgs, T_MeV, Ye):
   
   with h5py.File(nulib_path, "r") as f:
        #grid from NuLib_SFHo.h5
        E   = np.array(f["neutrino_energies"])   # [18] MeV
        R   = np.array(f["rho_points"])          # [82] g/cc
        Tgs = np.array(f["temp_points"])         # [65] MeV
        Y   = np.array(f["ye_points"])           # [51]
    
        
        # --- ADD: expose bin edges and ΔE^3 weights as globals ---
        global bin_widths, bin_bottom, bin_top, E_mid, W3  

        bin_widths = np.array(f["bin_widths"], dtype=float)    # [N]
        bin_bottom = np.array(f["bin_bottom"], dtype=float)    # [N] MeV
        bin_top = np.array(f["bin_top"],    dtype=float)    # [N] MeV
        E_mid = E   # alias for clarity
        W3 = bin_top**3 - bin_bottom**3   # MeV^3



        interpolator= True

        if interpolator is False:

          ir = _nearest_idx(R, rho_cgs)
          iT = _nearest_idx(Tgs, T_MeV)
          iY = _nearest_idx(Y, Ye)

          print(f"[NuLib] Actual table point at rho={R[ir]:.3e} g/cc, T={Tgs[iT]:.2f} MeV, Ye={Y[iY]:.2f}")


          # tables: [E, species(3), Ye, T, rho]
          kabs = f["absorption_opacity"]          # (18,3,51,65,82)

          i_nue, i_anue, i_nux = 0, 1, 2
        
          Gam_nue  = kabs[:, i_nue,  iY, iT, ir]
          Gam_anue = kabs[:, i_anue, iY, iT, ir]
          Gam_nux  = kabs[:, i_nux,  iY, iT, ir]
          Gam_anux = Gam_nux
        
        else:
          i_nue, i_anue, i_nux = 0, 1, 2

          Gam_nue  = np.array([interpolate_eas(ig, i_nue,  rho_cgs, T_MeV, Ye, f, "absorption_opacity") for ig in range(len(E))])
          
          Gam_anue  = np.array([interpolate_eas(ig, i_anue,  rho_cgs, T_MeV, Ye, f, "absorption_opacity") for ig in range(len(E))])

          Gam_nux  = np.array([interpolate_eas(ig, i_nux,  rho_cgs, T_MeV, Ye, f, "absorption_opacity") for ig in range(len(E))])
                              
        Gam_anux=Gam_nux
                              
        Gam_nu= 0.5*(Gam_nue + Gam_nux)
        Gam_anu= 0.5*(Gam_anue + Gam_anux)

        return E, Gam_nu, Gam_anu



def scan_and_plot_imomega_vs_gbar(E, T_nue, T_anue, g_nue, Gam_nu, Gam_anu, target):
   
   gmin=0.1
   gmax=1.6
   npts=500

   g_list = np.linspace(gmin, gmax, npts)
   im_list = np.full_like(g_list, np.nan, dtype=float)
   ok_list = np.zeros_like(g_list, dtype=bool)

   f_nux  = np.zeros_like(E)
   f_anux = np.zeros_like(E)

   w0 = 0.0 + 1e-6j
   
   print("CSV,g_anue,ImOmega_cm_inv,success")

   for i, gbar in enumerate(g_list):
     f_nue  = FD_dist(E, g=g_nue, T=T_nue)
     f_anue = FD_dist(E, g=gbar,  T=T_anue)
                                   
     df_nu  = f_nue  - f_nux
     df_anu = f_anue - f_anux

     w, info = solve_dispersion_twofluids(E, df_nu, Gam_nu, df_anu, Gam_anu, target, w0=w0)

     if (info.success):
       im_list[i] = w.imag
       ok_list[i] = True

     #  print(f"CSV,{gbar:.6f},{w.imag:.10e},1")


      
     #  w0 = w  # warm start next solve
     else:
       print("unsolved point, gbar=",gbar)

      # print(f"CSV,{gbar:.6f},nan,0")
       pass

   plt.figure()
   plt.plot(g_list, im_list)
   plt.axhline(0.0, linestyle="--")
   plt.xlabel("g_anue")
   plt.ylabel("Im(omega) [cm$^{-1}$]")
   ttl_branch = "I = -1" if target == -1.0 else "I = 3"
   plt.title(f"hCFI growth vs g_anue ({ttl_branch})")
   plt.tight_layout()
   plt.savefig("imomega_vs_gbar.png", dpi=200, bbox_inches="tight")
   plt.close()
   print("Saved plot to imomega_vs_gbar.png")

   n_ok = int(ok_list.sum())
   print(f"Solved {n_ok}/{len(g_list)} points. "
         f"Max Im(w) = {np.nanmax(im_list):.3e} cm^-1 at g_anue ≈ {g_list[np.nanargmax(im_list)]:.3f}"
                   )


def get_nulib_ranges(nulib_path: str):

   """Return (rho_min, rho_max, T_min, T_max, Ye_min, Ye_max) from the NuLib table."""
   with h5py.File(nulib_path, "r") as f:
     rho_tab = np.array(f["rho_points"],  dtype=float)
     T_tab   = np.array(f["temp_points"], dtype=float)
     Ye_tab  = np.array(f["ye_points"],   dtype=float)
     
     return (float(rho_tab.min()), float(rho_tab.max()), float(T_tab.min()), float(T_tab.max()), float(Ye_tab.min()), float(Ye_tab.max()))




def validate_existence(rho: float, T: float, Ye: float, rho_min: float, rho_max: float,
T_min: float,   T_max: float,Ye_min: float,  Ye_max: float,where: str = ""):
   
   # NaN/Inf check
   if not (np.isfinite(rho) and np.isfinite(T) and np.isfinite(Ye)):
     #print(f"[error] {where} has non-finite state: rho={rho}, T={T}, Ye={Ye}")
     #sys.exit(1)
     raise ValueError(f"[error] {where} has non-finite state: rho={rho}, T={T}, Ye={Ye}")

   # NuLib range check
   if not (rho_min <= rho <= rho_max):
     #print(f"[error] {where} rho={rho} out of NuLib range [{rho_min}, {rho_max}]")
     #sys.exit(1)
     raise ValueError(f"[error] {where} rho={rho} out of NuLib range [{rho_min}, {rho_max}]")

   if not (T_min <= T <= T_max):
     #print(f"[error] {where} T={T} MeV out of NuLib range [{T_min}, {T_max}]")
     #sys.exit(1)
     raise ValueError(f"[error] {where} rho={rho} out of NuLib range [{T_min}, {T_max}]")

   if not (Ye_min <= Ye <= Ye_max):
     #print(f"[error] {where} Ye={Ye} out of NuLib range [{Ye_min}, {Ye_max}]")
     #sys.exit(1)
     raise ValueError(f"[error] {where} rho={rho} out of NuLib range [{Ye_min}, {Ye_max}]")

def avg_gam(E, Gam_nu, Gam_anu, df_nu, df_anu):
  
   Gam_nu_av= (np.sum((E**2)*df_nu*Gam_nu* bin_widths))/(np.sum((E**2)*df_nu* bin_widths))

   Gam_anu_av=  (np.sum((E**2)*df_anu*Gam_anu* bin_widths))/(np.sum((E**2)*df_anu* bin_widths))          

   return Gam_nu_av, Gam_anu_av             



def solve_single_point(idx: Tuple[int,int,int], mono: bool, opac: bool, branch): 
   
   bracket_fail_e = 0
   bracket_fail_a = 0
   bracket_fail_x = 0


   model_path = "model_rl1_orthonormal.h5"
   voxel = idx #set the index for (x,y,z)
   
   global x_cm, y_cm, z_cm

   n_e, n_a, n_x, u_e, rho_cgs, T_MeV, Ye, x_cm, y_cm, z_cm = load_model_nJ(model_path, "e", voxel, include_state=True) 
   
   if verbose > 0:
     print(f"rho= {rho_cgs:.6e} g/cc, T_MeV= {T_MeV: .6e} MeV, Y= {Ye: .6e}")

   # ---- EOS point for opacities (choose point to be tested) ----
   #rho_cgs = 1.0e11   # g/cc
   #T_MeV   = 10.0      # MeV
   #Ye      = 0.20
   

   # ---- path to nulib table ----
   if mod_op:
      nulib_path = "NuLib_SFHo_corrected.h5"
   else:  
      nulib_path = "NuLib_SFHo_rho82_temp65_ye51_ng16_ns3_version1.0_20210311 1.h5"


   # get ranges from file
   rho_min, rho_max, T_min, T_max, Ye_min, Ye_max = get_nulib_ranges(nulib_path)

   validate_existence(rho_cgs, T_MeV, Ye, rho_min, rho_max, T_min, T_max, Ye_min, Ye_max, where=f"voxel {voxel}")
    
   #opac= True


   if(opac==True):
     E, Gam_nu, Gam_anu = load_gamma_from_nulib(nulib_path, rho_cgs, T_MeV, Ye)
     #print(f"[NuLib] Using table at rho={rho_cgs:.3e} g/cc, T={T_MeV:.2f} MeV, Ye={Ye:.2f}")
     
     if verbose > 0:
       print(f"[state] voxel={voxel} rho={rho_cgs:.3e} g/cc  T={T_MeV:.3f} MeV  Ye={Ye:.3f}")

     T_nue= T_MeV #MeV
     T_anue= T_MeV


   else:

     #bin_widths=None

     E_max=200
     N=200

     E=np.linspace(0.0,E_max,N)
   
     E0=10.0
     gamma0 = 1e-5  # scaling equation factor
     Gam_nu = gamma0 * (np.where(E>0, E, 0)/E0)**2
     Gam_anu = gamma0 * (np.where(E>0, E, 0)/E0)**2
     
     T_nue= 4.0 #MeV
     T_anue= 5.0

    # parameters
   
   #trial_model= False
   
   if trial_model is True:
     g_nue= 1.0
     g_anue= 0.6 #placeholder value. It is supposed to be a free variable and we will scan it.

     f_nue= FD_dist(E, g_nue, T_nue)

     f_nux= np.zeros_like(E)

     f_anue= FD_dist(E, g_anue, T_anue)
   
     f_anux= np.zeros_like(E)

     df_nu = f_nue - f_nux 

     df_anu = f_anue - f_anux
   
   else:

     X_IS_PAIR = True
     # Fit spectra from table n,J on the NuLib grid for the same voxel
     theta_e, A_e, _, fE_e, bracket_fail_e = spectral_dist_fit(model_path, "e", voxel)
     
     if bracket_fail_e == 1:
       print(f"Bracketing failed for 'e' at voxel={voxel} rho={rho_cgs:.3e} g/cc  T={T_MeV:.3f} MeV  Ye={Ye:.3f}")  

     theta_a, A_a, _, fE_a, bracket_fail_a = spectral_dist_fit(model_path, "a", voxel)   

     if bracket_fail_a == 1:
       print(f"Bracketing failed for 'a' at voxel={voxel} rho={rho_cgs:.3e} g/cc  T={T_MeV:.3f} MeV  Ye={Ye:.3f}")

     # --- add heavy–lepton spectrum fit ---
     theta_x, A_x_tot, _, fE_x_tot, bracket_fail_x = spectral_dist_fit(model_path, "x", voxel)
     
     if bracket_fail_x == 1:
       print(f"Bracketing failed for 'x' at voxel={voxel} rho={rho_cgs:.3e} g/cc  T={T_MeV:.3f} MeV  Ye={Ye:.3f}")

     # For any code that expects FD_dist(g, T): set g=A and T=theta
     g_nue, T_nue  = A_e, theta_e
     g_anue, T_anue = A_a, theta_a 

     f_nue, f_anue = fE_e, fE_a
     
     if X_IS_PAIR:
       # split the total x population evenly between ν_x and \barν_x
       f_nux  = 0.25 * fE_x_tot
       f_anux = 0.25 * fE_x_tot

       if verbose > 0:
         print(f"[spectrum x] voxel={voxel}  kBT_x={theta_x:.3f} MeV, A_x_total={A_x_tot:.3e}  (split 25/25)")
     else:
       # treat n_x, J_x as per-species already
       f_nux  = fE_x_tot
       f_anux = fE_x_tot

       if verbose > 0:
         print(f"[spectrum x] voxel={voxel}  kBT_x={theta_x:.3f} MeV, A_x={A_x_tot:.3e}")     
     #f_nux  = np.zeros_like(E)
     #f_anux = np.zeros_like(E)
     df_nu  = f_nue  - f_nux
     df_anu = f_anue - f_anux
     
     if verbose > 0:
       print(f"[spectrum fit] voxel={voxel}  kBT_e={theta_e:.3f} MeV, A_e={A_e:.3e} | kBT_a={theta_a:.3f} MeV, A_a={A_a:.3e}")

 #  w_test = 0.0 + 1e-6j   # tiny positive imaginary seed, in cm^-1
 #  Itest = I_omega_twofluids(w_test, E, df_nu, Gam_nu, df_anu, Gam_anu)
 #  print("I_twofluids(w_test) =", Itest)  # should be a finite complex number (dimensionless)
   
   if mono is True :
     
     Gam_nu_av, Gam_anu_av= avg_gam(E, Gam_nu, Gam_anu, df_nu, df_anu)
     
     #im_om_p= ((Gam_nu_av - Gam_anu_av)/2.0) * np.abs((n_e + n_a - 0.5*n_x)/(n_e - n_a)) - ((Gam_nu_av + Gam_anu_av)/2.0)

     #im_om_m= -((Gam_nu_av - Gam_anu_av)/2.0) * np.abs((n_e + n_a - 0.5*n_x)/(n_e - n_a)) - ((Gam_nu_av + Gam_anu_av)/2.0)
     
     im_om_p= (Gam_nu_av*( n_a - 0.25*n_x ) - Gam_anu_av*( n_e - 0.25*n_x ))/ ( n_e - n_a )

     im_om_m= -1.0*(Gam_nu_av*( n_e - 0.25*n_x ) - Gam_anu_av*( n_a - 0.25*n_x ))/ ( n_e - n_a )

     if im_om_p > im_om_m and np.isfinite(im_om_p):
       im_om= im_om_p
     else:
       im_om= im_om_m


     return im_om, rho_cgs, T_MeV, Ye
       

   one_fluid= False

   if(one_fluid==True):
     # solve without anti_nu: set target ---
     w0 = 0.0 + 1e-3j
     omega, info = solve_dispersion(E, df_nu, Gam_nu, target=branch, w0=w0)

     print("omega =", omega)
     print("success:", info.success, "|", info.message)

     I_w = I_omega(omega, E, df_nu, Gam_nu)
     print("I(omega)[verificastion]=","",I_w)


   # --- two-fluid solve: set target ---
   w0 = 0.0 + 1e-6j  #tiny imaginary seed, in cm^-1
   omega2, info2 = solve_dispersion_twofluids(E, df_nu, Gam_nu, df_anu, Gam_anu, target=branch, w0=w0)
   

   if verbose > 0:
     print("omega[2-fluid] =", omega2)
     print("success:", info2.success, "|", info2.message)
     print("I_twofluids(omega) check =", I_omega_twofluids(omega2, E, df_nu, Gamma_nu, df_anu, Gamma_anu))

   
   if trial_model is True:
     scan_and_plot_imomega_vs_gbar(E, T_nue, T_anue, g_nue, Gam_nu, Gam_anu, -1.0)
   
   return omega2, info2, rho_cgs, T_MeV, Ye


def point():
  

   # ---- EOS point for opacities (choose point to be tested) ----

   T_Mev  = 5.043395403786411  #MeV
   rho_cgs  = 128058601870.03752  #g/cc
   Ye  = 0.1563186433420338

   E = np.array([
    1, 3, 5.2382, 8.0097, 11.442, 15.691, 20.953, 27.468,
    35.536, 45.525, 57.895, 73.212, 92.178])  #MeV

   energybinstopMeV = np.array([
    2, 4, 6.4765, 9.543, 13.34, 18.042, 23.864, 31.073,
    39.999, 51.052, 64.738, 81.685, 102.67])

   energybinsbottomMeV = np.array([
    0, 2, 4, 6.4765, 9.543, 13.34, 18.042, 23.864, 31.073,
    39.999, 51.052, 64.738, 81.685])
   
   Gam_nu_e =  np.array([1.1279864349828499e-08, 4.7428025628186796e-08, 
                        1.321984832248114e-07, 3.294158114604995e-07, 7.528396092747342e-07,
                        1.5644452508683583e-06, 2.9487331095815796e-06,
                        5.1520838790595784e-06, 8.605493196949606e-06, 
                        1.4043114431085709e-05, 2.262451577257334e-05, 
                        3.6135649577238473e-05, 5.731900449884377e-05 ]) #cm^-1
   
   Gam_nu_x = Gam_anu_x=  np.array([2.174662492044255e-10, 3.0391841308327543e-10,
                        4.466580398128411e-10, 6.371159236091845e-10,
                        8.969420647822444e-10, 1.2650799494987814e-09, 
                        1.777602214007199e-09, 2.3794705306483863e-09, 
                        3.149801121595646e-09, 4.208531959563822e-09, 
                        5.307878395583215e-09, 6.9122806344757855e-09, 
                        8.655465405273294e-09])
   
   Gam_anu_e =  np.array([1.3566096423813977e-09, 4.4755275717913815e-09,
                           2.1546754963019424e-08, 5.47531006400814e-08,
                           1.1123883054511091e-07, 2.042363205567264e-07, 
                           3.5482012940187117e-07, 5.920655243201202e-07, 
                           9.526436748501559e-07, 1.4809548374270669e-06, 
                           2.2289900021493824e-06, 3.2538736229147686e-06, 
                           4.6123319631062746e-06])
   
   Gam_nu= 0.5*(Gam_nu_e + Gam_nu_x)

   Gam_anu= 0.5*(Gam_anu_e + Gam_anu_x)

   Gam_nu_s= c*Gam_nu # s^-1
   Gam_anu_s= c*Gam_anu 

   # Neutrino distribution functions

   f_nue = np.array([
    5.3245656748033138e-03, 1.5978206640139703e-02, 2.7340626736756864e-02,
    3.2089846556718205e-02, 2.6210613940471663e-02, 1.6053556351649751e-02,
    7.4581054456212296e-03, 2.5160179998209029e-03, 5.8403718278747475e-04,
    8.7777097910152665e-05, 7.9876592427094722e-06, 4.0330773860229255e-07,
    9.9487753023125056e-09
    ])
   f_anue = np.array([
    7.9745741660033973e-04, 2.0650218959726886e-03, 6.8306989510108339e-03,
    1.0342781961943959e-02, 1.0473633341442235e-02, 7.7470351075407829e-03,
    4.2587128928068020e-03, 1.7132291811786122e-03, 4.8448195133447586e-04,
    9.0279294456845565e-05, 1.0161726609247737e-05, 6.1963359675856731e-07,
    1.7935548751120651e-08
    ])
   f_nux = np.array([
    1.9122703099187557e-04, 2.7041688466178338e-04, 2.9470926976743635e-04,
    2.6194434096463122e-04, 1.9219021924125727e-04, 1.1498074419462892e-04,
    5.4858517739206768e-05, 1.9913514618089655e-05, 5.3637149245166015e-06,
    1.0168916876718995e-06, 1.2490438261083011e-07, 9.4997316882871965e-09,
    4.0860253082545327e-10
    ])
   


   #f_nue = np.array([
   # 7.0994208997377517e-03, 1.6569992071255989e-02, 2.7850408881147370e-02,
   # 3.2482205465428897e-02, 2.6448855470892776e-02, 1.6173687053825551e-02,
   # 7.5060897551031645e-03, 2.5305516364041926e-03, 5.8710787249954973e-04,
   # 8.8210207810425468e-05, 8.0248563560583814e-06, 4.0510307523564364e-07,
   # 9.9916360277372577e-09
   # ])

   #f_anue = np.array([
   # 1.0632765554671197e-03, 2.1415041884161211e-03, 6.9580613700387338e-03,
   # 1.0469241982139077e-02, 1.0568833493640193e-02, 7.8050071074430823e-03,
   # 4.2861128000531027e-03, 1.7231255532653416e-03, 4.8702920994658761e-04,
   # 9.0724750699417242e-05, 1.0209047968987736e-05, 6.2239191451207997e-07,
   # 1.8012817621610205e-08
   # ])

   #f_nux = np.array([
   # 2.5496937465583415e-04, 2.8043232483444201e-04, 3.0020429828160784e-04,
   # 2.6514710466691839e-04, 1.9393713337566092e-04, 1.1584115899831680e-04,
   # 5.5211469049980155e-05, 2.0028543916201174e-05, 5.3919157047453655e-06,
   # 1.0219092363027128e-06, 1.2548604018241358e-07, 9.5420200320866388e-09,
   # 4.1036284808555850e-10
   #])
   
   f_anux= f_nux

   df_nu = f_nue - f_nux 

   df_anu = f_anue - f_anux

   

   # --- multi ---
   w0 = 0.0 + 1e-6j  #tiny imaginary seed, in cm^-1
   #w0_s= 0.0 + 1e5j # initial guess in s^-1
   #denom_nu  = w0 + 1j * Gam_nu
   #denom_anu = w0 + 1j * Gam_anu

   global bin_widths

   bin_widths= np.abs(energybinstopMeV-energybinsbottomMeV)

   #print(f'bin_widths={bin_widths}')

   #term_nu  = np.sum((E**2) * df_nu  / denom_nu * bin_widths)
   #term_anu = np.sum((E**2) * df_anu / denom_anu * bin_widths)
   
   #Iw= C_PREFAC * (term_nu - term_anu)
   
   def F1(xy):
     target= -1.0
     wr, wi = xy
     w = complex(wr, wi)
     Iw = I_omega_twofluids(w, E, df_nu, Gam_nu, df_anu, Gam_anu)
     return [Iw.real - target, Iw.imag]

   sol = root(F1, [w0.real, w0.imag], method="hybr")
   omega = sol.x[0] + 1j*sol.x[1]

   print(f"multi_succes={sol.success}")
   print(f"omega={omega}")
   omega_s=omega*c
   print(f"omega_multi_s^-1={omega_s: .2e} s^-1")

   I_check= I_omega_twofluids(omega, E, df_nu, Gam_nu, df_anu, Gam_anu)

   print(f"I_check={I_check}")
   #--mono--
   
   luke=True

   if luke:
     
     Gam_nu_e_av= (np.sum((E**2)*f_nue*Gam_nu_e* bin_widths))/(np.sum((E**2)*f_nue* bin_widths))*c #1/s
     Gam_anu_e_av= (np.sum((E**2)*f_anue*Gam_anu_e* bin_widths))/(np.sum((E**2)*f_anue* bin_widths))*c #1/s
     Gam_nu_x_av= (np.sum((E**2)*f_nux*Gam_nu_x* bin_widths))/(np.sum((E**2)*f_nux* bin_widths))*c #1/s
     Gam_anu_x_av= (np.sum((E**2)*f_anux*Gam_anu_x* bin_widths))/(np.sum((E**2)*f_anux* bin_widths))*c #1/s

     Gam_nu_av= 0.5*(Gam_nu_e_av + Gam_nu_x_av)
     Gam_anu_av= 0.5*(Gam_anu_e_av + Gam_anu_x_av)



     #Gam_nu_av= (np.sum((E**2)*df_nu*Gam_nu_s* bin_widths))/(np.sum((E**2)*df_nu* bin_widths))*c #1/s

     #Gam_anu_av=  (np.sum((E**2)*df_anu*Gam_anu_s* bin_widths))/(np.sum((E**2)*df_anu* bin_widths))*c #1/s
                          
     w3= np.abs(energybinstopMeV**3-energybinsbottomMeV**3)

     n_e= (C_f/3.0)*np.sum(f_nue*w3)

     n_a= (C_f/3.0)*np.sum(f_anue*w3)

     n_x= (C_f/3.0)*np.sum(f_nux*w3)

    

     #print(f"n_e={n_e}")
     #print(f"n_a={n_a}")
     #print(f"n_x={n_x}")

     gamma = (Gam_nu_av + Gam_anu_av) / 2.0 # 1/s
     alpha = (Gam_nu_av - Gam_anu_av) / 2.0 # 1/s

     g    = GF_cm3_per_s*np.sqrt(2.0)*(n_e - n_x)
     gbar = GF_cm3_per_s*np.sqrt(2.0)*(n_a - n_x)
     
     
     
     G = (g + gbar)/2.0 
     A = (g - gbar)/2.0 
     
     #print(f"G={G: .2e}")
     #print(f"A={A: .2e}")
     #print(f"alpha={alpha: .2e}")
     #print(f"gamma={gamma: .2e}")

     if A*A >= np.abs(G*alpha):

       print("here")  
       max_im_omega_pres  = -gamma + np.abs(G*alpha/A)
       max_im_omega_break = -gamma + np.abs(G*alpha/A)

       print(f"mono_max_im_omega_pres={max_im_omega_pres: .2e}") 
       print(f"mono_max_im_omega_break={max_im_omega_break: .2e}") 

       return 0

     else:

       max_im_omega_pres  = -gamma + np.sqrt(np.abs(G*alpha))
       max_im_omega_break = -gamma + np.sqrt(np.abs(G*alpha))/np.sqrt(3)

       print(f"mono_max_im_omega_pres={max_im_omega_pres: .2e}") 
       print(f"mono_max_im_omega_break={max_im_omega_break: .2e}") 

       return 0

   Gam_nu_av= (np.sum((E**2)*df_nu*Gam_nu* bin_widths))/(np.sum((E**2)*df_nu* bin_widths))

   Gam_anu_av=  (np.sum((E**2)*df_anu*Gam_anu* bin_widths))/(np.sum((E**2)*df_anu* bin_widths))  

   w3= np.abs(energybinstopMeV**3-energybinsbottomMeV**3)

   n_e= (C_f/3.0)*np.sum(f_nue*w3)

   n_a= (C_f/3.0)*np.sum(f_anue*w3)

   n_x= (C_f/3.0)*np.sum(f_nux*w3)


   im_om_p= (Gam_nu_av*( n_a - n_x ) - Gam_anu_av*( n_e - n_x ))/ ( n_e - n_a )

   im_om_m= -1.0*(Gam_nu_av*( n_e - n_x ) - Gam_anu_av*( n_a - n_x ))/ ( n_e - n_a )

   #if im_om_p > im_om_m and np.isfinite(im_om_p):
   #  im_om= im_om_p
   #else:
   #  im_om= im_om_m
   
   print(f"mono_omega_imag_p={im_om_p}")
   print(f"mono_omega_imag_m={im_om_m}")
   
   print(f"mono_omega_imag_p={im_om_p*c: .2e} s^-1")
   print(f"mono_omega_imag_m={im_om_m*c : .2e} s^-1")
   
   return 0 


if __name__ == "__main__":

   if ("--scan-slice" in sys.argv):
     
     mono= False
     model_path = "model_rl1_orthonormal.h5"  
     axis = "y"      # 'x', 'y', or 'z'
     index = 128
     stride = 1
     branch = -1.0   # or 3.0 for the other branch
     opac=True
      
     if ("--mono" in sys.argv):
       mono= True
     #trial_model= False

     for arg in sys.argv:
       if arg.startswith("--axis="):   axis       = arg.split("=",1)[1]
       if arg.startswith("--index="):  index      = int(arg.split("=",1)[1])
       if arg.startswith("--stride="):  stride      = int(arg.split("=",1)[1])

     if mono is True:
       out_csv = f"omega_slice_{axis}{index}_stride{stride}_mono.csv" 
     else:  
       if mod_op:
          out_csv = f"omega_slice_{axis}{index}_mod_branch{branch}.csv"
       else:
          out_csv = f"omega_slice_{axis}{index}_stride{stride}_branch{branch}.csv"   
   
     with h5py.File(model_path, "r") as _f:
       nx, ny, nz = _f["rho(g|ccm)"].shape
     
     if axis == "x":
       if not (0 <= index < nx): raise IndexError(f"x index {index} out of bounds [0,{nx})")
       loop_i = range(0, ny, stride); loop_j = range(0, nz, stride)
       

     elif axis == "y":
       if not (0 <= index < ny): raise IndexError(f"y index {index} out of bounds [0,{ny})")
       loop_i = range(0, nx, stride); loop_j = range(0, nz, stride)
       

     elif axis == "z":
       if not (0 <= index < nz): raise IndexError(f"z index {index} out of bounds [0,{nz})")
       loop_i = range(0, nx, stride); loop_j = range(0, ny, stride)
       

     else:
       raise ValueError("axis must be 'x', 'y', or 'z'")

     #SR rho = modelfile["rho(g|ccm)"][:,index,:]
     #SR imomega = np.zeros(len(loop_i), len(loop_j))
     #SR [[fill the 2D array quantities]]
     #SR outfile = h5py.File(filename,"w")
     #SR outfile["rho(g|ccm)"] = rho
     #SR outfile["imomega(1|cm)"] = imomega
     #SR outfile.close()

     with open(out_csv, "w", newline="") as fp:
       w = csv.writer(fp)
       w.writerow(["ix","iy","iz","omega_real_cm^-1","omega_imag_cm^-1","success","rho[g/cc]","T[MeV]","Ye","x(cm)","y(cm)","z(cm)"])

       pos_im_cm = 0
       pos_vox  = None
     
       for i in loop_i:
         for j in loop_j:
             
           if axis == "x":
             voxel = (index, i, j)
           elif axis == "y":
             voxel = (i, index, j)
           else:  # axis == "z"
             voxel = (i, j, index)
             
           try:
             
             if mono is True:
               im_om, rho_cgs, T_MeV, Ye = solve_single_point(voxel, mono, opac, branch)
               im_cm_inv = im_om
               r_cm_inv = 0.0
               success= True
               

             else:
               omega, info, rho_cgs, T_MeV, Ye = solve_single_point(voxel, mono, opac, branch) 
               im_cm_inv = float(omega.imag)
               r_cm_inv = float (omega.real)
               success= info.success

             w.writerow([voxel[0], voxel[1], voxel[2], float(r_cm_inv), im_cm_inv, success, rho_cgs, T_MeV, Ye, x_cm, y_cm, z_cm])
            
             if np.isfinite(im_cm_inv) and im_cm_inv > 0:
               pos_im_cm = im_cm_inv
               pos_vox  = (voxel, rho_cgs, T_MeV, Ye)
               (ix,iy,iz), rho_p, T_p, Ye_p = pos_vox
               print(f" positive Im(omega) = {pos_im_cm:.3e} cm^-1 at ({ix},{iy},{iz}) " f"with rho={rho_p:.3e} g/cc, T={T_p:.3f} MeV, Ye={Ye_p:.3f}")
               


           except Exception:
                                     
             w.writerow([voxel[0], voxel[1], voxel[2], np.nan, np.nan, False, np.nan, np.nan, np.nan, x_cm, y_cm, z_cm])


     print(f"[scan-slice] wrote {out_csv}")

     #if best_im_cm > 0:
     #  (ix,iy,iz), rho_b, T_b, Ye_b = best_vox
     #  print(f"[scan-slice] max Im(omega) ≈ {best_im_cm:.3e} cm^-1 at ({ix},{iy},{iz}) " f"with rho={rho_b:.3e} g/cc, T={T_b:.3f} MeV, Ye={Ye_b:.3f}")
   

   elif("--point" in sys.argv):

     s= point()      