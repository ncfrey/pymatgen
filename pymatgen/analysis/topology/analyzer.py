# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import warnings
import numpy as np
import os
import logging

from monty.json import MSONable

import z2pack

"""
This module makes extensive use of the z2pack tool for calculating topological invariants to identify topological phases. It is mainly meant to be used in band structure workflows for high throughput classification of band topology.

If you use this module, please cite the following papers:

Dominik Gresch, Gabriel Autès, Oleg V. Yazyev, Matthias Troyer, David Vanderbilt, B. Andrei Bernevig, and Alexey A. Soluyanov “Z2Pack: Numerical Implementation of Hybrid Wannier Centers for Identifying Topological Materials” [PhysRevB.95.075146]

Alexey A. Soluyanov and David Vanderbilt “Computing topological invariants without inversion symmetry” [PhysRevB.83.235401]

"""

__author__ = "Nathan C. Frey"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Nathan C. Frey"
__email__ = "ncfrey@lbl.gov"
__status__ = "Development"
__date__ = "July 2019"


class BandTopologyAnalyzer:
    def __init__(self, input_dir="input", surface=lambda t1, t2: [t1, t2, 0]):
        """A class for analyzing band structure topology and diagnosing non-trivial topological phases.

        Create a z2pack.fp.System instance for vasp and Wannier90 that points to inputs and allows for dynamic calling of vasp.

        When called from a root directory, input files (POSCAR, INCAR, etc.) must be in a folder called 'input'.

        Examples of possible Brillouin zone surfaces:
        [0, t1 / 2, t2]  : k_x = 0
        [1/2, t1 /2 , t2]  : k_x = 1/2

        Required VASP flags:
            LWANNIER90 = .TRUE.
            LWRITE_MMN_AMN = .TRUE.
            ISYM = -1
            NBANDS = (set to divisible by num of cores so no extra bands are added)

        Required Wannier90 flags:
            exclude_bands = (set to exclude unoccupied bands)

        Args:
            input_dir (str): Path to input vasp and Wannier90 input files.
            surface (function): Brillouin zone surface, defaults to (kx, ky, 0).

        Parameters:
            system (z2pack System object): Configuration for dynamically calling vasp within z2pack.
            surface (lambda function): Parameterizes surface in Brillouin zone.

        """

        # Create a Brillouin zone surface for calculating the Wilson loop / Wannier charge centers (defaults to k_z = 0 surface)
        self.surface = surface

        # Define input file locations
        input_files = ["CHGCAR", "INCAR", "POSCAR", "POTCAR", "wannier90.win"]
        input_files = [input_dir + "/" + s for s in input_files]

        # Create k-point inputs for VASP
        kpt_fct = z2pack.fp.kpoint.vasp

        system = z2pack.fp.System(
            input_files=input_files,
            kpt_fct=kpt_fct,
            kpt_path="KPOINTS",
            command="srun vasp_std >& log",
            mmn_path="wannier90.mmn",
        )

        self.system = system

    def run(self, z2_settings=None):
        """Calculate Wannier charge centers on the BZ surface.

        Args:
            z2_settings (dict): Optional settings for specifying convergence criteria. Check z2_defaults for keywords.

        """

        system = self.system
        surface = self.surface

        # z2 calculation defaults
        z2d = {
            "pos_tol": 0.01,  # change in Wannier charge center pos
            "gap_tol": 0.3,  # Limit for closeness of lines on surface
            "move_tol": 0.3,  # Movement of WCC between neighbor lines
            "num_lines": 11,  # Min num of lines to calculate
            "min_neighbour_dist": 0.01,  # Min dist between lines
            "iterator": range(8, 27, 2),  # Num of kpts to iterate over
            "load": True,  # Start from most recent calc
            "save_file": "z2run.json",
        }  # Serialize results

        # User defined setting updates to defaults
        if z2_settings:
            for k, v in z2_settings.items():
                d = {k: v}
                z2d.update(d)

        # Calculate WCC on the Brillouin zone surface.
        result = z2pack.surface.run(
            system=system,
            surface=surface,
            pos_tol=z2d["pos_tol"],
            gap_tol=z2d["gap_tol"],
            move_tol=z2d["move_tol"],
            num_lines=z2d["num_lines"],
            min_neighbour_dist=z2d["min_neighbour_dist"],
            iterator=z2d["iterator"],
            save_file=z2d["save_file"],
        )

        self.output = BandTopologyAnalyzerOutput(result, surface)


class BandTopologyAnalyzerOutput(MSONable):
    def __init__(self, result, surface, chern_number=None, z2_invariant=None):
        """
        Class for storing results of band topology analysis.

        Args:
            result (object): Output from z2pack.surface.run()
            surface (list): BZ surface parameterization.
            chern_number (int): Chern number.
            z2_invariant (int): Z2 invariant. 
            
        """

        self._result = result
        self.surface = surface
        self.chern_number = chern_number
        self.z2_invariant = z2_invariant

        self._parse_result(self, result)

    def _parse_result(self, result, surface):

        # Topological invariants
        chern_number = z2pack.invariant.chern(result)
        z2_invariant = z2pack.invariant.z2(result)

        self.chern_number = chern_number
        self.z2_invariant = z2_invariant

        # BZ surface as a vector
        surface_vec = surface("t1", "t2")
        self.surface = surface_vec
