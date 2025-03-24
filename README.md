# Primerize (NA_Thermo)

<img src="https://primerize.stanford.edu/site_media/images/logo_primerize.png" alt="Primerize Logo" width="200" align="right">

My fork of Primerize meant to add degenerate primer functionality. The thermodynamic and dynamic programming logic for the Primderize_1D class has been updated to allow for input and processing of sequencing containing degenerate codons. Goal is to prevent degenerate codons from being in regions of overlap.

Also a few updates for python3 functionality and package management

Currently minimal tests or error handling, use at your own risk. Primerize_2D and 3D have not been changed or tested.

Please view [Original Repository](https://github.com/ribokit/Primerize)


## Installation

To install:

```bash
# Download Package
git clone https://github.com/nkoranda97/Primerize_Degenerate
# Navigate to Package
cd Primerize_Degenerate
# Create venv
python -m venv venv # name whatever you want
# Activate venv
source venv/bin/activate
# Install Package
pip install .

```

#### Test

To test if **Primerize** is functioning properly in local installation, run the *unit test* scripts:

```bash
cd path/to/Primerize/tests/
python -m unittest discover
```

All test cases should pass.


## Usage

For simple Primer Design tasks, follow this example:

```python

from primerize_degen.primerize_1d import Primerize_1D

worker: Primerize_1D = Primerize_1D()

worker.set("MIN_LENGTH", 10)
worker.set("MAX_LENGTH", 65)
SEQUENCE: str = """CCCTGCGTGAAGCTGACCAACACCTCCACACTGACTCAGGCTTGTCCCAAGGTGACATTCGACCCTATTCCAATCCACTACTGCGCTCCTGCAGGCTATGCCATCCTGAAATGTAACAATAAGACCTTTAACGGCAAAGGGCCATGCAACAATGTGAGCACTGTCCAGTGTACCCACGGCATCAAGCCCGTGGTCTCAACACAGCTGCTGCTGAACGGGAGCCTGGCAGAGGAAGAGATTGTGATCAGATCAAAAAACCTGAGGAACNNKNNKNNKATCATTATCGTGCAGCTGAATAAGAGTGTGGAGATCGTCTGCACACGACCTAACAATGGC"""

job = worker.design(sequence=SEQUENCE)


for i, primer in enumerate(job.primer_set, start=1):
    print(f"Test Primer {i},{primer}\n")
```

README updated 2025-03-24
