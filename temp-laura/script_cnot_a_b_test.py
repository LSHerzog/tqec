import os
import sys
from datetime import datetime
from pathlib import Path

import tqec.computation.syndrome_extraction_cc as se
from tqec import Basis, NoiseModel
from tqec.computation.pipe_prism import PrismPipeKind
from tqec.computation.prism import BasisPrism, Position3DHex
from tqec.computation.prism_graph import PrismGraph
from tqec.gallery.cnot import cnot
from tqec.simulation.plotting.inset import plot_observable_as_inset
from tqec.simulation.simulation import start_simulation_using_sinter
from tqec.simulation.split import split_stats_for_observables

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../teleportation_superdense"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import correlated_matching_helper
import matplotlib.pyplot as plt
import numpy as np
import sinter

from tqec.computation.correlation import (
    CorrelationSurface,
    ZXEdge,
    ZXNode,
)
from tqec.interop.pyzx.plot import draw_correlation_surface_on, draw_positioned_zx_graph_on
from tqec.utils.enums import Basis

tele_type = "a_b"  #!ADAPT FOR CORRECT PATHS

# =============================color code================================

g = PrismGraph("CNOT")

prisms = [
    (Position3DHex(3, 3, 0), "XN", "control-int"),
    (Position3DHex(3, 3, 1), "NN", ""),
    (Position3DHex(3, 3, 2), "NN", ""),
    (Position3DHex(3, 3, 3), "NX", "control-out"),
    (Position3DHex(2, 2, 1), "XN", ""),
    (Position3DHex(2, 2, 2), "NZ", ""),
    (Position3DHex(1, 3, 0), "XN", "target-in"),
    (Position3DHex(1, 3, 1), "NN", ""),
    (Position3DHex(1, 3, 2), "NN", ""),
    (Position3DHex(1, 3, 3), "NX", "target-out"),
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

# CORRELATION SRUFACE CC
idx = 1
correlation_surfaces = g.find_correlation_surfaces()
cs_cc_0 = correlation_surfaces[idx]

cs_cc_1 = CorrelationSurface(
    frozenset(
        {
            ZXEdge(
                ZXNode(6, Basis.X),
                ZXNode(7, Basis.X),
            ),
            ZXEdge(
                ZXNode(7, Basis.X),
                ZXNode(8, Basis.X),
            ),
            ZXEdge(
                ZXNode(8, Basis.X),
                ZXNode(9, Basis.X),
            ),
        }
    )
)

cs_cc_lst = [cs_cc_0, cs_cc_1]

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
g_sc = cnot(Basis.X)
correlation_surfaces = g_sc.find_correlation_surfaces()

idx = 1

cs_sc_0 = correlation_surfaces[idx]
zx_sc = g_sc.to_zx_graph()

cs_sc_1 = CorrelationSurface(
    frozenset(
        {
            ZXEdge(
                ZXNode(6, Basis.X),
                ZXNode(7, Basis.X),
            ),
            ZXEdge(
                ZXNode(7, Basis.X),
                ZXNode(8, Basis.X),
            ),
            ZXEdge(
                ZXNode(8, Basis.X),
                ZXNode(9, Basis.X),
            ),
        }
    )
)

cs_sc_lst = [cs_sc_0, cs_sc_1]


# plot SC CS
for cs_idx, cs_sc in enumerate(cs_sc_lst):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_positioned_zx_graph_on(zx_sc, ax, node_size=100)
    draw_correlation_surface_on(cs_sc, zx_sc, ax)
    fig.savefig(f"correlation_surface_zx_sc_{tele_type}_{cs_idx}.pdf")

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")


if __name__ == "__main__":
    d_lst = [3, 5]
    ks = [(d - 1) / 2 for d in d_lst]
    circuit_builders = [se.SyndromeExtractionStimCC(g, d=d) for d in d_lst]

    add_missing_detectors = False

    p_values = list(np.logspace(-3, -1, 10))

    rounds = d_lst

    num_workers = 16

    max_shots = 200  # 4_000_000#100_000#10_000#100_000#10_000#10_000_000

    max_errors = 10  # 400#100#500

    se_type = "run_all_superdense"

    path_cc = (
        f"cnot_sinter_{tele_type}_{timestamp}_maxshots{max_shots}_maxerrors{max_errors}_cc.csv"
    )

    path_sc = f"cnot_sinter_{tele_type}_{timestamp}_maxshots{max_shots}_maxerrors{max_errors}_sc_correlated.csv"
    database_path_sc = f"cnot_sinter_{tele_type}_{timestamp}_maxshots{max_shots}_maxerrors{max_errors}_sc_correlated.pkl"

    stats_sc = start_simulation_using_sinter(
        g_sc,
        ks=ks,
        ps=p_values,
        noise_model_factory=NoiseModel.uniform_depolarizing,
        manhattan_radius=2,
        observables=cs_sc_lst,
        decoders=["correlated_pymatching"],
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        save_resume_filepath=Path(path_sc),
        database_path=Path(database_path_sc),
        custom_decoders={
            "correlated_pymatching": correlated_matching_helper.CorrelatedPyMatchingDecoder()
        },
        split_observable_stats=True,  #!important if multiple observables are joined.
    )

    print("SC FINISHED, STARTING CC...")

    stats_cc = se.run_experiment_sinter(
        circuit_builders=circuit_builders,
        rounds=rounds,
        p_values=p_values,
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        path=path_cc,
        add_missing_detectors=add_missing_detectors,
        se_type=se_type,
        cs_lst=cs_cc_lst,  #!within the function we fixed count_observable_error_combos=True
    )

    # -----------------------RELOAD-----------------------
    stats_cc = sinter.read_stats_from_csv_files(path_cc)
    stats_sc = sinter.read_stats_from_csv_files(path_sc)

    assert len(cs_sc_lst) == len(cs_cc_lst), "same operators have to be applied on both codes."
    n_obs = len(cs_cc_lst)

    # -----------------------PLOTTING----------------------

    for s in stats_sc:
        print(s.json_metadata, "shots:", s.shots, "errors:", s.errors, "discards:", s.discards)

    split_cc = split_stats_for_observables(
        stats_cc, num_observables=n_obs
    )  # split_cc[obs_idx] = list[TaskStats]
    split_sc = split_stats_for_observables(stats_sc, num_observables=n_obs)

    for obs_idx in range(n_obs):
        fig, ax = plt.subplots()

        sinter.plot_error_rate(
            ax=ax,
            stats=split_sc[obs_idx],
            x_func=lambda s: s.json_metadata["p"],
            group_func=lambda s: f"Surface Code d={s.json_metadata['d']}",
            plot_args_func=lambda *_: {"linestyle": ":"},
        )

        sinter.plot_error_rate(
            ax=ax,
            stats=split_cc[obs_idx],
            x_func=lambda s: s.json_metadata["p"],
            group_func=lambda s: f"Color Code d={s.json_metadata['d']}",
            plot_args_func=lambda *_: {"linestyle": "-"},
        )

        plot_observable_as_inset(ax, zx_cc, cs_cc_lst[obs_idx])

        ax.loglog()
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
        ax.set_xlabel("Physical Error Rate")
        ax.set_ylabel("Logical Error Rate")
        # ax.set_xlim(1e-3, 1e-1)
        # ax.set_ylim(1e-5, 0.999)  #! adjust for other experiments
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))

        fig.savefig(
            f"cnot_error_rate_{tele_type}_{timestamp}_obs{obs_idx}_maxerrors{max_errors}_maxshots{max_shots}.pdf"
        )
        plt.close(fig)
