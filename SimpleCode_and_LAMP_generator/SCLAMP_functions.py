from pathlib import Path
from multiprocessing import Pool
import os, time

"""
The user specifies a casesPy.txt file. For each handle in it, this script will: 
	Create LAMP input file.
	Run LAMP to generate the output files.
	Not repeat already generated files.
All "default" values that are not specified by the handles can be changed in this script.
"""


def parse_handle(handle):
    uList = [
        pos for pos, char in enumerate(handle) if char == "_"
    ]  # list of underscore indices
    uList.append(handle.rfind("-"))
    height1 = float(handle[uList[0] + 2 : uList[1]])
    period1 = float(handle[uList[1] + 2 : uList[2]])
    angle1 = float(handle[uList[2] + 2 : uList[3]])
    height2 = float(handle[uList[3] + 3 : uList[4]])
    period2 = float(handle[uList[4] + 3 : uList[5]])
    angle2 = float(handle[uList[5] + 3 : uList[6]])
    shipspeed = float(handle[uList[6] + 2 : uList[7]])
    realizationIndex = handle[uList[7] + 1 :]
    return (
        height1,
        period1,
        angle1,
        height2,
        period2,
        angle2,
        shipspeed,
        realizationIndex,
    )


def create_LAMP_input_file(LAMPhandle):
    """Step 1: Check to see if the appropriate input file already exists
    Step 2: parse the SC handle to get values of significant wave height, period, speed, etc.
    Step 3: create the input file. This is the section where other default values can be found."""
    path = Path("LAMP_files\\" + LAMPhandle + ".in")
    if path.is_file():
        print(LAMPhandle + " already has an input file.")
        return  # Step 1
    (
        height1,
        period1,
        angle1,
        height2,
        period2,
        angle2,
        shipspeed,
        realizationIndex,
    ) = parse_handle(
        LAMPhandle
    )  # Step 2
    # Step 3 below
    with open("LAMP_files\\" + LAMPhandle + ".in", "w") as f:
        # the [:-8] is to exclude the realization index from the input filename, per the norm for SC
        f.write(
            f"""!01 DESCR - Descriptive Title (max 80 char)
ONR Topsides Study - Flared variant
!02 FPROG - Source file for programmer's input (blank for defaults)

!03 FAPLT - Source file for autopilot input (blank for defaults)

!04 FGEOM - Source file for geometry definition
topsides_fl.lmp
!05 FOUT - Destination file for primary output
LAMP_files\\{LAMPhandle}.out
!06 Output frequency for pressure, geometry, etc.
!  POUT    GOUT    SOUT    BOUT
	0 0 0   0 0 0   0 0 0   0 0 0
!07 FPOUT - File for pressure data output

!08 FGOUT - File for geometry output

!09 FSOUT - File for balance check output

!10 FBOUT - File for elastic beam output

!11 AUXOUT(20) - Flags for auxiliary output files
	0 0 0 0 0   0 0 0 0 0   0 0 0 0 0   0 0 0 0 0
!12 IVEC - Use Non-vectorized (0) or vectorized (1) kernel
	0
!13 LMPTYP (1-4)  MOTYPE (forced/impulsive/free)  MIXED (Rankine/mixed/IRF)
	2     2    -1
!13 (cont) IFSMIX NXFS NYFS  IMSMIX NXMS NZMS - mixed-source surface points
	5    51     26     1    21    8
!13 (cont) XMIX  YMIX  ZMIX - mixed-source surface extent
	450.000      225.0000      20.0
!14 TINIT NSTEP DTH IRST - Initial Time, Number of Steps, Time Step, Restart
	0.0   18000  0.1    0
!15 USHIP  UCURNT  DCURNT  WDEPTH - Steady speed, current vel/dir, water depth
	{shipspeed*.5114444}     0.0   0.0   0.0   
!16 PMGIN(1:6) Initial position and orientation in global frame
	0.00000       0.00000       2.30000
	0.00000       0.00000       0.00000
!17 VMGSHP(1:6) Initial Velocity and Rotation rate in ship fixed system
	{shipspeed*.5114444}    0.0   0.0  
	0.00000       0.00000       0.00000
!18 AMPM(1:6) Amplitude for forced sinusoidal motion (if MOTYPE=0)
	0.00000       0.00000       0.00000
	0.00000       0.00000       0.00000
!19 OMEGM(1:6) Frequency for forced sinusoidal motion (if MOTYPE=0)
	0.00000       0.00000       0.00000
	0.00000       0.00000       0.00000
!20 SWTCH(1:6) Sets which modes of motion will be considered
0     0     1
1     1     0
!21 ISEA  NWAVES  NWSC
!      0
!21 ISEA  NWAVES  NWSC
!      1      1      2
!21 (cont) FREQW  PHASEW  AMPW  HEADW = component wave data
!           0.5   0.0    1.0   180.0
!21 ISEA SIGWHT TMODAL SEAHD SPREAD NFREQ NHEAD IWVRLZ NWSC
2   {height1}   {period1}   {angle1}    0.0    120    1    {realizationIndex}    2
!21 (cont) WSCSTP  WSCFAC = wave scaling data
0   0.00000
100   1.00000
!22 GRAVIN  RHOIN  LENIN  ANGIN - Scale Factors for Input
9.807       1.025       154.000       57.2958
!23 GRAVOUT  RHOOUT  LENOUT  ANGOUT - Scale Factors for Output
9.807       1.025       154.000       57.2958
!24 GSHIFT(3), GORIG(3), GROT(3) - Input geometry transformation
79.53688       0.00000       0.00000
0.00000       0.00000       0.00000
0.00000       0.00000       0.00000
!25 SMA - Ship mass, SMI(1,1),(2,1),(3,1),(1,2)...(3,3) Mom. of Inertia
1.00000
56.55        0.00000        0.00000
0.00000       1383.0        0.00000
0.00000       0.00000       1383.0
!26 RGRAV - center of gravity in input system
!  0.00000       0.00000       2.306    !  limit  GM = 2.01m
!  0.00000       0.00000       3.316    !  small  GM = 1.00m
!  0.00000       0.00000       2.816    !  medium GM = 1.50m
0.00000       0.00000       2.000    !  KG =7.5 m
!27 SYMGEO= 1 for Symmetry in calc., SYMINP =1 for symmetry in input
0     1
!28 NCOMP0 ...
10
!28 (cont) KCTYPE0 NEWWL0 KSPWL0 SPWL0 NEWST0 KSPST0 SPST0,1->NCOMP0
0     0     1   0.00000         0     1   0.00000    ! hull bulb
0     0     1   0.00000         0     1   0.00000    ! hull fwd
0     0     1   0.00000         0     1   0.00000    ! hull aft
0     0     1   0.00000         0     1   0.00000    ! skeg side
0     0     1   0.00000         0     1   0.00000    ! skeg bottom
0     0     1   0.00000         0     1   0.00000    ! topsides bow
0     0     1   0.00000         0     1   0.00000    ! topsides mid
0     0     1   0.00000         0     1   0.00000    ! topsides aft
1     0     1   0.00000         0     1   0.00000    ! deck
1     0     1   0.00000         0     1   0.00000    ! transom

!29 IVM, IHM, ITM, NBCA, NBMX ...
!     0    0    0     0     0
$INCLUDE:LMP_ONRFL_loads_rigid.inp
!30 KINVIS (kinematic viscosity)
	0.118800E-05
!30 (cont)  IHROLL (roll damping option)
	4
!30 (cont) CRITDAMP - ratio of linear critical damping coefficient
	0.10000 
!31 IHLIFT (hull lift option)
	0
!32 NFIN - Number of wing-like lifting appendages (e.g. rudder, fins)
	0
!33 NBK - Number of plate-like lifting appendages (e.g. bilge keels)
0
 
"""
        )
        if (
            height2 > 0.01
        ):  # this section is for adding a secondary spectrum (bimodal seas)
            f.write(
                f"""$ADDSEA:
!2nd (additional) seaway definition:
! Short-reseted SS6 defined by Bretschneider spectrum
!ADDSEA01 ISEA - Seaway option
	2
!ADDSEA01 (cont) SIGWHT TMODAL (Bretschneider Spectrum)
	{height2}      {period2}
!ADDSEA01 (cont) SEAHD SPREAD NFREQ NHEAD IWVRLZ
	{angle2}   0.0   120    1     {realizationIndex}
!ADDSEA02 SPEC_FREQ(2) - frequency range for discretizing wave spectrum
	0.00000 	0.00000
 
"""
            )
        f.write(
            """!$DAMPING:
!Supplemental damping
!DAMPING01 IVSDMP - Supplemental damping option
!     1
!DAMPING02 VSDMP_COEFF(6) - Linear damping coefficients
!  0.0e0
!  0.0e0
!  1.4166E+4 ! Heave; SimpleCode DampHeave=0.5 (A33=0.0)
!  0.0e0     ! Roll specified above as CRITDAMP
!  3.4332E+5 ! Pitch; SimpleCode DampPitch=0.5 (A55=0.0)
!  0.0e0
 
$OPTIONS:
!IDYNA - Dynamic solver option
 IDYNA 0
!IEKZOPT  - option for computing exponential incident wave decay term
 IEKZOPT    0
!MASSDISTOPT - mass distribution matching option
 MASSDISTOPT 0  ! use input mass distribution as is"""
        )


def run_LAMP_input_file(LAMPhandle):
    path = Path(LAMPhandle + ".mot")
    if path.is_file():
        return  # Step 1
    _, _, _, _, _, _, _, realizationIndex = parse_handle(LAMPhandle)  # Step 2
    command = "lamp.exe LAMP_files\\" + LAMPhandle
    os.system(command)
    command2 = "lmplot.exe -s get_motion.scpt -root LAMP_files\\" + LAMPhandle
    os.system(command2)


"""
The user specifies a casesPy.txt file. For each handle in it, this script will: 
	Create SimpleCode input file.
	Run SimpleCode to generate the output files.
	Not repeat already generated files.
All "default" values that are not specified by the handles can be changed in this script.
"""


def parse_handle(handle):
    uList = [
        pos for pos, char in enumerate(handle) if char == "_"
    ]  # list of underscore indices
    uList.append(handle.rfind("-"))
    height1 = float(handle[uList[0] + 2 : uList[1]])
    period1 = float(handle[uList[1] + 2 : uList[2]])
    angle1 = float(handle[uList[2] + 2 : uList[3]])
    height2 = float(handle[uList[3] + 3 : uList[4]])
    period2 = float(handle[uList[4] + 3 : uList[5]])
    angle2 = float(handle[uList[5] + 3 : uList[6]])
    shipspeed = float(handle[uList[6] + 2 : uList[7]])
    realizationIndex = handle[uList[7] + 1 :]
    return (
        height1,
        period1,
        angle1,
        height2,
        period2,
        angle2,
        shipspeed,
        realizationIndex,
    )


def create_SC_input_file(SChandle):
    """Step 1: Check to see if the appropriate input file already exists
    Step 2: parse the SC handle to get values of significant wave height, period, speed, etc.
    Step 3: create the input file. This is the section where other default values can be found."""
    path = Path("SimpleCode_files//" + SChandle[:-8] + ".in")
    if path.is_file():
        print("Input file " + SChandle[:-8] + ".in" + " already exists.")
        return  # Step 1
    (
        height1,
        period1,
        angle1,
        height2,
        period2,
        angle2,
        shipspeed,
        realizationIndex,
    ) = parse_handle(
        SChandle
    )  # Step 2
    # Step 3 below
    with open("SimpleCode_files//" + SChandle[:-8] + ".in", "w") as f:
        # the [:-8] is to exclude the realization index from the input filename, per the norm for SC
        f.write(
            "$OFFSETS:flh.off\n"
            "\n"
            "$Simulation:\n"
            "	1          ! SimType  - simulation type\n"
            "!   Simulation:\n"
            "!              1 - heave/roll/pitch\n"
            "!              2 - surge/sway/yaw\n"
            "!              3 - adhoc 6DOF\n"
            "!              4 - true 6DOF\n"
            "!   Wave pass/forces:\n"
            "!             -1 - irregular, balanced\n"
            "!             -2 - irregular, fixed\n"
            "!   Wave pass/GZ :\n"
            "!             -3 - irregular, fixed\n"
            "!             -4 - irregular, balaning: GZCALC\n"
            "!\n"
            "	18000       ! Npt  - # of time steps\n"  # Change this to influence time steps.  Originally 18000
            "	2000        ! Nrec - # of records (realizations)\n"
            "	0.1        ! st   - time step size (sec)\n"
            "	-1          ! HF_fade - high-frequency attenuation factor (integer)\n"
            "	3          ! IGZW   - roll restoring option (1=scaled GZ,3=volume)\n"
            "	0.0        ! rollmx - maximum roll angle (deg)\n"
            "	0.1        ! StW  - time step for incident wave evaluation (sec)\n"
            "	0.0        ! RampTime  - Time for incident wave ramp (sec)\n"
            "	0.0        ! WaveLenAttn - Length for short-wave HS+FK attenuation\n"
            "	6          ! output_mot - motion output option\n"
            "!              0 - none\n"
            "!              1 - 6-DOF motion with heave relative to static position\n"
            "!              2 - old style\n"
            "!              3-  old stile (unformatted)\n"
            "!              4-  6DOF + wave elevation and slope\n"
            "!              5 - LAMP style\n"
            "!              6 - LAMP stile + wave elevation and slope\n"
            "    0         ! output_vel - velocity output option\n"
            "!              0 -none\n"
            "!              1-6-DOF velocity; translational velocity in global system\n"
            "!              2 - old-style 3-DOF extended output (inc. vel and acc)\n"
            "!              3 - 6-DOF velocity, translational velocity in semi-fixed system\n"
            "!              4 - 6-DOF velocity, translational velocity in ship-fixed system\n"
            "    0          ! output_sea - seaway output option\n"
            "!              0 -none\n"
            "!              1 - ANIM seaway file\n"
            "!              2 - old-style SimpleCode .frq file\n"
            "    0          ! output_hs - hydrostatics output option\n"
            "!              0 -none\n"
            "!              1 - Hydrostatics/dynamics summary and calm water GZ curve\n"
            "!              2 - also HS curves (.hsc)\n"
            "!              3 - also debug HS curves (.Dhsc)\n"
            "    0          ! output_frc - force output option\n"
            "!              0 -none\n"
            "!              1 - 3-DOF force/moment in global coordinates ala LAMP\n"
            "!              2 - 6-DOF force/moment in global coordinates ala LAMP\n"
            "\n"
            "\n"
            "$Dynamics:\n"
            "! KG : Vertical position of center of gravity wrt baseline\n"
            "  7.5\n"
            "! DampHeav,DampRoll,DampPitch - Damping as fracion of critical\n"
            "  0.50000      0.104      0.50000\n"
            "! rix,riy,riz - Radii of gyration wrt beam or length\n"
            "  0.40000     0.2415       0.2415\n"
            "! A33,A44,A55 - Added mass coefficients\n"
            "  1.00000      0.211       1.0000\n"
            "! RollInv - Roll angle to use inverted damping\n"
            "   0.0000\n"
            "! DampRollInv - Roll Damping in capsized position\n"
            "   0.5000\n"
            "\n"
            "$Initial:\n"
            "! Tinit  - initial time (sec)\n"
            "  0.0\n"
            "! Rinit  - initial position (m) and orientation (deg)\n"
            "  0.0 0.0 0.0\n"
            "  0.0 0.0 0.0\n"
            "! Vinit  - initial velocity (m/s) and rotation rates (deg/s)\n"
            "  0.0 0.0 0.0\n"
            "  0.0 0.0 0.0\n"
            "\n"
            "$Control:\n"
            "! SpeedCntrlOpt - Speed control option (future use)\n"
            "   0\n"
            "! VsKn - Target/specified forward speed in knots\n"
            f"   {shipspeed}\n"
            "! HeadCntrlOpt - Heading control option (future use)\n"
            "   0\n"
            "! HeadDeg -Target/specified heading angle in degrees\n"
            "  0.0 \n"
            "\n"
            "$Seaway_Spectrum:\n"
            "! ISPEC - Spectrum option\n"
            "   2\n"
            "! Bretschneider two parameter spectrum\n"
            "! SigWaveHght  ModalPeriod\n"
            f"    {height1}          {period1}\n"
            "!SeaHead  Spread  Nfreq  Nhead\n"
            f"   {angle1}      0      120     1  \n"
            "\n"
        )
        if height2 > 0.01:
            f.write(
                "$Seaway_Spectrum:\n"
                "! ISPEC - Spectrum option\n"
                "   2\n"
                "! Bretschneider two parameter spectrum\n"
                "! SigWaveHght  ModalPeriod\n"
                f"    {height2}          {period2}\n"
                "!SeaHead  Spread  Nfreq  Nhead\n"
                f"   {angle2}      0      120     1  \n"
                "\n"
            )

        f.write("$Loads:fl154_loads.inp")


def run_SC_input_file(SChandle):
    path = Path(SChandle + ".mot")
    if path.is_file():
        print("Output file " + SChandle + ".mot already exists")
        return  # skipping cases for which .mot already exists
    _, _, _, _, _, _, _, realizationIndex = parse_handle(SChandle)  # Step 2
    command = (
        "simplecode "
        + "SimpleCode_files\\"
        + SChandle[:-8]
        + " "
        + realizationIndex
        + " 1"
    )
    os.system(command)
