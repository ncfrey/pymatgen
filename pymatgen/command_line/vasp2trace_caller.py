# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import subprocess
import warnings
import numpy as np
import pandas as pd

from os import path
import os

from monty.json import MSONable
from monty.dev import requires
from monty.os.path import which

"""
This module implements an interface to the vasp2trace code for generating a text file that can be used to analyze band topology.

This module depends on the vasp2trace script available in the path.
Please download at http://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl and consult the README.pdf for further help.

If you use this module, please cite the following:
    
M.G. Vergniory, L. Elcoro, C. Felser, N. Regnault, B.A. Bernevig, Z. Wang Nature(2019) 566, 480-485. doi:10.1038/s41586-019-0954-4 
"""

__author__ = "ncfrey"
__version__ = "0.1"
__maintainer__ = "Nathan C. Frey"
__email__ = "ncfrey@lbl.gov"
__status__ = "Development"
__date__ = "July 2019"

VASP2TRACEEXE = which("vasp2trace")


class Vasp2TraceCaller:
    @requires(
        VASP2TRACEEXE,
        "Vasp2TraceCaller requires vasp2trace to be in the path."
        "Please follow the instructions at http://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl.",
    )
    def __init__(self, folder_name):

        """
        Run vasp2trace to find the set of irreducible representations at each maximal k-vec of a space group, given the eigenvalues.

        vasp2trace requires a self-consistent VASP run with the flags ISTART=0 and ICHARG=2; followed by a band structure calculation with ICHARG=11 and LWAVE=.True.

        High-symmetry kpts that must be included in the band structure path for a given spacegroup can be found in the max_KPOINTS_VASP folder in the vasp2trace directory.
        
        Args:
            folder_name (str): Path to directory with OUTCAR and WAVECAR of band structure run with wavefunctions at the high-symmetry kpts.

        """

        # Check for OUTCAR and WAVECAR
        if not path.isfile(folder_name + "/OUTCAR") or not path.isfile(
            folder_name + "/WAVECAR"
        ):
            raise FileNotFoundError()

        # Call vasp2trace
        os.chdir(folder_name)
        process = subprocess.Popen(
            ["vasp2trace"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        stdout = stdout.decode()

        if stderr:
            stderr = stderr.decode()
            warnings.warn(stderr)

        if process.returncode != 0:
            raise RuntimeError(
                "vasp2trace exited with return code {}.".format(process.returncode)
            )

        self._stdout = stdout
        self._stderr = stderr

        # Process output
        self.output = Vasp2TraceOutput(stdout)


class Vasp2TraceOutput(MSONable):
    def __init__(
        self,
        vasp2trace_stdout,
        num_occ_bands=None,
        soc=None,
        num_symm_ops=None,
        symm_ops=None,
        num_max_kvec=None,
        kvecs=None,
        num_kvec_symm_ops=None,
        symm_ops_in_little_cogroup=None,
        traces=None,
    ):
        """
        This class processes results from vasp2trace to classify material band topology and give topological invariants.
        
        Refer to http://www.cryst.ehu.es/html/cryst/topological/File_Description.txt for further explanation of parameters.

        Args:
            vasp2trace_stdout (txt file): stdout from running vasp2trace.
            num_occ_bands (int): Number of occupied bands.
            soc (int): 0: no spin-orbit, 1: yes spin-orbit
            num_symm_ops (int): Number of symmetry operations.
            symm_ops (list): Each row is a symmetry operation (with spinor components if soc is enabled)
            num_max_kvec (int): Number of maximal k-vectors.
            kvecs (list): Each row is a k-vector.
            num_kvec_symm_ops (dict): {kvec_index: number of symm operations in the little cogroup of the kvec}. 
            symm_ops_in_little_cogroup (dict): {kvec_index: int indices that correspond to symm_ops}
            traces (dict): band index, band degeneracy, energy eigenval, Re eigenval, Im eigenval for each symm op in the little cogroup 
            
        """

        self._vasp2trace_stdout = vasp2trace_stdout

        self.num_occ_bands = num_occ_bands
        self.soc = soc
        self.num_symm_ops = num_symm_ops
        self.symm_ops = symm_ops
        self.num_max_kvec = num_max_kvec
        self.kvecs = kvecs
        self.num_kvec_symm_ops = num_kvec_symm_ops
        self.symm_ops_in_little_cogroup = symm_ops_in_little_cogroup
        self.traces = traces

        self._parse_stdout(vasp2trace_stdout)

    def _parse_stdout(self, vasp2trace_stdout):

        with open(vasp2trace_stdout, "r") as file:
            lines = file.readlines()

            # Get header info
            num_occ_bands = int(lines[0])
            soc = int(lines[1])  # No: 0, Yes: 1
            num_symm_ops = int(lines[2])
            symm_ops = np.loadtxt(lines[3 : 3 + num_symm_ops])
            num_max_kvec = int(lines[3 + num_symm_ops])
            kvecs = np.loadtxt(
                lines[4 + num_symm_ops : 4 + num_symm_ops + num_max_kvec]
            )

            # Dicts with kvec index as keys
            num_kvec_symm_ops = {}
            symm_ops_in_little_cogroup = {}
            traces = {}

            # Start of trace info
            trace_start = 5 + num_max_kvec + num_symm_ops
            start_block = 0  # Start of this block

            # Block start line #s
            block_starts = []
            for jdx, line in enumerate(lines[trace_start - 1 :], trace_start - 1):
                # Parse input lines
                line = [i for i in line.split(" ") if i]
                if len(line) == 1:  # A single entry <-> new block
                    block_starts.append(jdx)

            # Loop over blocks of kvec data
            for idx, kpt in enumerate(kvecs):

                start_block = block_starts[idx]
                if idx < num_max_kvec - 1:
                    next_block = block_starts[idx + 1]
                    trace_str = lines[start_block + 2 : next_block]
                else:
                    trace_str = lines[start_block + 2 :]

                # Populate dicts
                num_kvec_symm_ops[idx] = int(lines[start_block])
                soilcg = [
                    int(i.strip("\n"))
                    for i in lines[start_block + 1].split(" ")
                    if i.strip("\n")
                ]
                symm_ops_in_little_cogroup[idx] = soilcg

                trace = np.loadtxt(trace_str)
                traces[idx] = trace

        # Set instance attributes
        self.num_occ_bands = num_occ_bands
        self.soc = soc
        self.num_symm_ops = num_symm_ops
        self.symm_ops = symm_ops
        self.num_max_kvec = num_max_kvec
        self.kvecs = kvecs
        self.num_kvec_symm_ops = num_kvec_symm_ops
        self.symm_ops_in_little_cogroup = symm_ops_in_little_cogroup
        self.traces = traces
