from datetime import datetime

import numpy as np

import tqec.computation.syndrome_extraction_cc as se
from tqec.computation.prism import Position3DHex
from tqec.computation.prism_graph import PrismGraph

g = PrismGraph("two-patches")

prisms = [
    (Position3DHex(0, 0, 0), "ZZ", "middle"),
    # (Position3DHex(1,1,0), "ZZ", "left"),
]

for pos, kind, label in prisms:
    g.add_prism(pos, kind, label)


timestamp = datetime.now().strftime("%y%m%d_%H%M%S")


if __name__ == "__main__":
    d_lst = [3, 5]
    circuit_builders = [se.SyndromeExtractionStimCC(g, d=d) for d in d_lst]

    p_values = list(np.round(np.geomspace(0.00001, 0.01, 8), 8))  # originally 0.0001

    rounds = d_lst

    num_workers = 4

    path = f"mem_sinter_{timestamp}.csv"

    stats = se.run_experiment_sinter(
        circuit_builders=circuit_builders,
        rounds=rounds,
        p_values=p_values,
        num_workers=num_workers,
        path=path,
    )
    # sinter.write_stats_to_csv("stats_tele.csv", stats)

    fig = se.plot_experiment_sinter(stats, d_lst)
    fig.savefig(f"LER_mem_{timestamp}.pdf")
