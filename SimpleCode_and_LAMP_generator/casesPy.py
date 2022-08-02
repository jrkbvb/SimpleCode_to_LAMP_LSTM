from pathlib import Path
from datetime import date

"""
Generates a text file descibing the experiement and containing a list of all the file handles (permutations) to be used to generate input files for simple code and Lamp.  Each handle contains:
- a prefix describing the experiement
- h: primary system wave height
- p: primary system period
- a: primary system angle relative to ship heading
- hh: secondary system
- pp: secondary system
- aa: secondary system

NOTE: The only items that need to be changed are 

in the format:
DESCRIPTION

***
SCprefix
LAMPprefix
_h##.#_p##.#_a##.#_hh##.#_pp##.#_aa##_s##.#-REALIZATION##
"""

# ------------------------ CHANGE THESE TO SET DESCRIPTION AND FILENAME -------------------------#
description = (
    f"This is my first experiment. This is test data.\nMy name\ncreated: "
    + str(date.today())
)  # year-month-day

# filename = "cases_demo_training.txt" #include .txt at the end
# filename = "cases_demo_validation.txt"
filename = "cases_demo_test.txt"

# ------------------------ CHANGE THESE TO ESTABLISH PREFIXES -------------------------#
prefix = ["SC", "LAMP"]
prefix = [string + "\n" for string in prefix]

# ------------------------ CHANGE THESE TO ESTABLISH PERMUTATION GENERATION -------------------------#
# Beam Seas only example
permutations = ""

# realizations = list(range(1,13))
# realizations = list(range(13, 17))
realizations = list(range(17, 21))

for r in realizations:
    r_string = "0" * (7 - len(str(r))) + str(r)
    permutations += "_h11.5_p16.4_a90_hh0_pp0_aa0_s0-" + r_string + "\n"

# ------------------------ DON'T CHANGE THESE -------------------------#
print(f"File description is:\n{description}")
print(f"\nThis will be save in:\n{filename}\n")
if Path("Cases_files\\" + filename).is_file():
    print("WARNING! A CASES FILE WITH THIS NAME ALREADY EXISTS!")
proceed = input("If you would like to proceed, enter '1':")
if proceed == "1":
    print("Writing file.")
    with open("Cases_files\\" + filename, "w") as f:
        f.write(description + "\n\n")  # Description
        f.write(3 * "*" + "\n")  # 3 *'s for marker
        f.writelines(prefix)  # prefix
        f.write(permutations)
else:
    print("Aborting.")
