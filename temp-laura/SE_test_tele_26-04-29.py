from datetime import datetime

import numpy as np
import sinter

import tqec.computation.syndrome_extraction_cc as se
from tqec.computation.pipe_prism import PrismPipeKind
from tqec.computation.prism import BasisPrism, Position3DHex
from tqec.computation.prism_graph import PrismGraph

g = PrismGraph("two-patches")

prisms = [
    (Position3DHex(0, 0, -1), "ZN", "init"),
    (Position3DHex(0, 0, 0), "NX", "left"),
    (Position3DHex(1, 1, 0), "XN", "right"),
    (Position3DHex(1, 1, 1), "NZ", "final"),
]

for pos, kind, label in prisms:
    g.add_prism(pos, kind, label)

pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[0][0], prisms[1][0], pipe_kind)

pipe_kind = PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z)
g.add_pipe(prisms[1][0], prisms[2][0], pipe_kind)

pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
g.add_pipe(prisms[2][0], prisms[3][0], pipe_kind)


timestamp = datetime.now().strftime("%y%m%d_%H%M%S")


if __name__ == "__main__":
    d_lst = [3, 5]
    circuit_builders = [se.SyndromeExtractionStimCC(g, d=d) for d in d_lst]

    add_missing_detectors = True

    p_values = list(np.round(np.geomspace(0.001, 0.05, 8), 8))  # originally 0.0001

    rounds = d_lst

    num_workers = 4

    path = f"tele_sinter_zexp-{timestamp}.csv"

    # stats = se.run_experiment_sinter(
    #    circuit_builders=circuit_builders,
    #    rounds=rounds,
    #    p_values=p_values,
    #    num_workers=num_workers,
    #    path = path,
    #    add_missing_detectors=add_missing_detectors
    # )
    # sinter.write_stats_to_csv("stats_tele.csv", stats)

    # fig = se.plot_experiment_sinter(stats, d_lst)
    # fig.savefig(f"LER_tele_zexp-{timestamp}.pdf")

    # reload and plot
    path = "tele_sinter_zexp-260505_092134.csv"
    stats = sinter.read_stats_from_csv_files(path)
    fig = se.plot_experiment_sinter(stats, d_lst)
    fig.savefig("LER_tele_zexp_reloaded.pdf")
