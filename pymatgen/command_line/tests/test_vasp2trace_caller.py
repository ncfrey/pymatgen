# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from pymatgen.command_line.vasp2trace_caller import Vasp2TraceCaller, Vasp2TraceOutput

from monty.os.path import which

import os
import numpy as np
import unittest

test_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "test_files/vasp2trace"
)

vasp2trace_cmd = which("vasp2trace")


class Vasp2TraceCallerTest(unittest.TestCase):
    def test_parsing(self):

        v2to = Vasp2TraceOutput(test_dir + "/Ag1Ge1Li2.txt")
        nob = v2to.num_occ_bands
        num_symm_ops = v2to.num_symm_ops
        symm_ops = v2to.symm_ops
        num_max_kvec = v2to.num_max_kvec
        kvecs = v2to.kvecs
        num_kvec_symm_ops = v2to.num_kvec_symm_ops
        symm_ops_in_little_cogroup = v2to.symm_ops_in_little_cogroup
        traces = v2to.traces

        self.assertEqual(nob, 68)
        self.assertEqual(num_symm_ops, 12)
        self.assertEqual(num_max_kvec, 4)
        self.assertEqual(num_kvec_symm_ops[0], 12)
        self.assertEqual(len(traces), 4)

        v2to = Vasp2TraceOutput(test_dir + "/Ba3Ca1O9Ru2.txt")
        nob = v2to.num_occ_bands
        num_symm_ops = v2to.num_symm_ops
        symm_ops = v2to.symm_ops
        num_max_kvec = v2to.num_max_kvec
        kvecs = v2to.kvecs
        num_kvec_symm_ops = v2to.num_kvec_symm_ops
        symm_ops_in_little_cogroup = v2to.symm_ops_in_little_cogroup
        traces = v2to.traces

        self.assertEqual(num_max_kvec, 8)
        self.assertEqual(len(traces), 8)


if __name__ == "__main__":
    unittest.main()
