from pathlib import Path
import os

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


# ------------------ Main Script --------------------#
casesFile = input("Enter name of casesPy.txt file: ")

file = open("Cases_files\\" + casesFile, "r")
lines = file.readlines()

foundStars = False
counter = 0
for line in lines:
    if foundStars == True:
        counter += 1
    if foundStars == True and counter == 1:
        if line[-1] == "\n":
            SCprefix = line[:-1]  # the -1 is to exclude the line break character (\n)
        else:
            SCprefix = line
    elif foundStars == True and counter >= 3:
        # This section is the main point of the script--> create input files and run SC
        if line[-1] == "\n":
            SChandle = (
                SCprefix + line[:-1]
            )  # again, to exclude the line break character
        else:
            SChandle = SCprefix + line
        create_SC_input_file(SChandle)
        run_SC_input_file(SChandle)

    elif line[:3] == "***":
        foundStars = True
