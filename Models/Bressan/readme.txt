%%%%%%%%%%%  last modified 23/12/2014  
%%%%%%%%%%%  new track files changed: added column PHASE see below
%%%%%%%%%%%  old files are in directory no_phase

PARSEC V1.2s tracks

Bressan A. et al. 2012, MNRAS, 427, 127 (V1.1)

Chen Y. et al. 2014, MNRAS, 444, 2525  (very low mass stars down to 0.1 Msun)
Tang J. et al. 2014, arXiv:1410.1745   (massive stars up to ~ 350Msun or more)
Chen Y. et al. 2014, to be submitted   (new bolometric corrections for massive stars)


From pre main sequence to   
- 30Gyr very low mass stars
- He flash   low mass stars
- a few thermal pulses intermediate mass stars
- C ignition massive stars

HB evolution  of low mass stars


Some filenames may contain the 'ADD' word.
Discard these files because they are used only to merge sections with 
different critical point numbers (see  the variable PHASE below).


For each track we provide a small table containing

MODELL MASS AGE 
LOG_L LOG_TE LOG_R (log10 Luminosity Teff Radius)
LOG_RAT (mass loss rate if not zero)
M_CORE_HE  (mass of H exhausted core  -M_sun-) 
M_CORE_C   (mass of He exhausted core  -M_sun-) 
H_CEN   HE_CEN  C_cen  O_cen (central composition by mass)
LX LY LC LNEUTR L_GRAV (in L/Ltot)
H_SUP  HE_SUP  C_SUP  N_SUP  O_SUP (surface composition by mass)
PHASE (added 23/12/2014) 



------------------------------------ PHASE ---------------------------------------
The last column (PHASE) represents the phase between selected critical points,  
along the tracks. 
Selected critical points for all tracks in each metallicity set
may be found within the corresponding files in  subdir 

pcrit

Do not use the second section after the line
# critical points in F7 files
because these critical points refer to the original files (not processed by the automatic isochrone machine).


In the track files, critical points correspond to integer values of the Phase, 
while fractional values are proportional to the fractional time duration 
between the current point and the beginning of that phase  
frac=(t-t_beg_i)/(t_end_i-t_beg_i))   (i=1, n_phases).

Integer values correspond to the following points along the tracks (see files in pcrit):
  1 PMS_BEG  Track begins here (Pre main sequence)
  2 PMS_MIN  
  3 PMS_END    PMS is near to end
  4 NEAR_ZAM   This point is very near the ZAMS
  5 MS_BEG     H burning fully active
  6 POINT_B    Almost end of the H burning. Small contraction phase begins here for interm. & massive stars   
  7 POINT_C    Small contraction ends here and star move toward RG
  8 RG_BASE    RG base
  9 RG_BMP1   RGB bump in Low Mass Stars (marked also for other masses)
 10 RG_BMP2   RGB bump end in Low Mass Stars (marked also for other masses)
 11 RG_TIP   Helium Flash or beginning of HELIUM Burning in intermediate and massive stars
 12 Loop_A   Base of Red He burning, before possible loop
 13 Loop_B   Bluest point of the loop (He burning
 14 Loop_C   central He = 0  almost 
 15 TPAGB begins or c_burning begins (massive stars, generally if LC > 2.9)

 
 Notes
 1) points are selected automatically and so there may be some approximate locations. 
 2) Not all tracks contain all phases (very low mass end when t=30Gyr, still on the main sequence) 
 3) Very massive stars have not clearly recognizable behaviour in the HR:
    He burning phases are mainly based on fractional central He content, plus last point (C-ignition)
    The same holds for HB tracks where only central He burning is present. 
	
	

