# using conda, I pip freeze > requirements_conda.txt which gives improper references for docker and
# most other ways to install from txt
#
# this script converts to interpretable requirements.txt

import re

# Open the original requirements file and the new cleaned file
with open("requirements_conda.txt", "r") as infile, open(
    "requirements_converted.txt", "w"
) as outfile:
    for line in infile:
        # Use regex to remove everything after the '@' symbol
        clean_line = re.sub(r" @ .*", "", line)
        outfile.write(clean_line)
