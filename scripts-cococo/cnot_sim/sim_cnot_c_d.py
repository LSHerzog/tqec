import os
import sys
from datetime import datetime

import tqec.computation.syndrome_extraction_cc as se
from tqec import Basis
from tqec.computation.pipe_prism import PrismPipeKind
from tqec.computation.prism import BasisPrism, Position3DHex
from tqec.computation.prism_graph import PrismGraph
from tqec.gallery.cnot import cnot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../teleportation_superdense"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import correlated_matching_helper
import matplotlib.pyplot as plt
import numpy as np
from sim_helper import run_and_plot

from tqec.interop.pyzx.plot import draw_correlation_surface_on, draw_positioned_zx_graph_on

tele_type = "c_d"  #!ADAPT FOR CORRECT PATHS

# =============================color code================================

g = PrismGraph("CNOT")

prisms = [
    (Position3DHex(3, 3, 0), "ZN", "control-int"),
    (Position3DHex(3, 3, 1), "NN", ""),
    (Position3DHex(3, 3, 2), "NN", ""),
    (Position3DHex(3, 3, 3), "NZ", "control-out"),
    (Position3DHex(2, 2, 1), "XN", ""),
    (Position3DHex(2, 2, 2), "NZ", ""),
    (Position3DHex(1, 3, 0), "ZN", "target-in"),
    (Position3DHex(1, 3, 1), "NN", ""),
    (Position3DHex(1, 3, 2), "NN", ""),
    (Position3DHex(1, 3, 3), "NZ", "target-out"),
]

for pos, kind, label in prisms:
    g.add_prism(pos, kind, label)

# c temp
pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[0][0], prisms[1][0], pipe_kind)

pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[1][0], prisms[2][0], pipe_kind)

pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[2][0], prisms[3][0], pipe_kind)

# t temp
pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[6][0], prisms[7][0], pipe_kind)

pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[7][0], prisms[8][0], pipe_kind)

pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[8][0], prisms[9][0], pipe_kind)

# spatial pipes
pipe_kind = PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z)
g.add_pipe(prisms[1][0], prisms[4][0], pipe_kind)

pipe_kind = PrismPipeKind(hor=BasisPrism.Z, ver=BasisPrism.X)
g.add_pipe(prisms[8][0], prisms[5][0], pipe_kind)

# middle temp
pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[4][0], prisms[5][0], pipe_kind)


cs_cc_lst = g.find_correlation_surfaces()
zx_cc = g.to_zx_graph()

# plot the CS in the ZX diagram
for cs_idx, cs_cc in enumerate(cs_cc_lst):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_positioned_zx_graph_on(zx_cc, ax, node_size=100)
    draw_correlation_surface_on(cs_cc, zx_cc, ax)
    fig.savefig(f"correlation_surface_zx_cc_{tele_type}_{cs_idx}.pdf")

"""
#run the circuit construction and distance computation:
for distance in [3,5,7]:
    rounds = distance
    p = 0.01
    SE = se.SyndromeExtractionStimCC(g, distance)
    circuit = SE.run_all_superdense(
        rounds,
        p,
        p,
        p,
        p,
        cs_lst = cs_cc_lst
    )
    d = len(circuit.shortest_graphlike_error())
    print("circuit level distance: ", d)
"""

# =============================surface code================================
g_sc = cnot(Basis.Z)
cs_sc_lst = g_sc.find_correlation_surfaces()

zx_sc = g_sc.to_zx_graph()


# plot SC CS
for cs_idx, cs_sc in enumerate(cs_sc_lst):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_positioned_zx_graph_on(zx_sc, ax, node_size=100)
    draw_correlation_surface_on(cs_sc, zx_sc, ax)
    fig.savefig(f"correlation_surface_zx_sc_{tele_type}_{cs_idx}.pdf")

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")


if __name__ == "__main__":
    d_lst = [3, 5, 7]
    ks = [(d - 1) / 2 for d in d_lst]
    circuit_builders = [se.SyndromeExtractionStimCC(g, d=d) for d in d_lst]

    add_missing_detectors = False

    p_values = list(np.logspace(-3, -1, 10))

    rounds = d_lst

    num_workers = 32

    max_shots = 4_000_000  # 100_000#10_000#100_000#10_000#10_000_000

    max_errors = 400  # 100#500

    se_type = "run_all_superdense"

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
