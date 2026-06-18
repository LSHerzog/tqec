import os
import sys
from datetime import datetime
from itertools import chain
from pathlib import Path

import tqec.computation.syndrome_extraction_cc as se
from tqec import BlockGraph, NoiseModel
from tqec.computation.correlation import find_correlation_surfaces
from tqec.computation.pipe_prism import PrismPipeKind
from tqec.computation.prism import BasisPrism, Position3DHex, ZXPrism
from tqec.computation.prism_graph import PrismGraph
from tqec.simulation.plotting.inset import plot_observable_as_inset
from tqec.simulation.simulation import start_simulation_using_sinter
from tqec.utils.position import Position3D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import correlated_matching_helper
from sim_helper import run_and_plot

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sinter
import temp_utils

tele_type = "a"

#=============================color code================================

g = PrismGraph("two-patches")

prisms = [
    (Position3DHex(0,0,-1), "ZN", "init"),
    (Position3DHex(0,0,0), "NX", "left"),
    (Position3DHex(1,1,0), "XN", "right"),
    (Position3DHex(1,1,1), "NZ", "final"),
]

for pos, kind, label in prisms:
    g.add_prism(pos, kind, label)

pipe_kind = PrismPipeKind(hor = BasisPrism.N, ver = BasisPrism.N)
g.add_pipe(prisms[0][0], prisms[1][0], pipe_kind)

pipe_kind = PrismPipeKind(hor = BasisPrism.X, ver = BasisPrism.Z)
g.add_pipe(prisms[1][0], prisms[2][0], pipe_kind)

pipe_kind = PrismPipeKind(hor = BasisPrism.N, ver = BasisPrism.N)
g.add_pipe(prisms[2][0], prisms[3][0], pipe_kind)

cs_cc_lst = g.find_correlation_surfaces()
zx_cc = g.to_zx_graph()

#=============================surface code================================
g_sc = BlockGraph("Teleportation")
cubes = [
    # Initial logical qubit
    (Position3D(0, 0, 0), "ZXZ", "Init_Z"),

    # Intermediate teleportation structure
    (Position3D(0, 0, 1), "ZXX", ""),
    (Position3D(0, 1, 1), "ZXX", ""),

    # Output logical qubit
    (Position3D(0, 1, 2), "ZXZ", "Out"),
]


for pos, kind, label in cubes:
    g_sc.add_cube(pos, kind, label)

pipes = [
    (0, 1),
    (1, 2),
    (2, 3),
]

for p0, p1 in pipes:
    g_sc.add_pipe(cubes[p0][0], cubes[p1][0])

cs_sc_lst = g_sc.find_correlation_surfaces()


timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

if __name__ == '__main__':

    zx = g.to_zx_graph()

    d_lst = [3,5,7]
    ks = [(d - 1) / 2 for d in d_lst]
    circuit_builders = [se.SyndromeExtractionStimCC(g, d=d) for d in d_lst]

    add_missing_detectors = False

    p_values = list(np.logspace(-4, -1, 10))

    rounds = d_lst

    num_workers = 32

    max_shots = 4_000_000#100_000#10_000#100_000#10_000#10_000_000

    max_errors = 400#100#500

    run_and_plot(
        g_cc=g,
        g_sc=g_sc,
        cs_cc_lst=cs_cc_lst,
        cs_sc_lst=cs_sc_lst,
        zx_cc=zx_cc,
        d_lst=d_lst,
        p_values=p_values,
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        tele_type=tele_type,
        timestamp=timestamp,
        correlated_decoder=correlated_matching_helper.CorrelatedPyMatchingDecoder(),
        run_simulation=True,  # set False to only reload and replot
    )

