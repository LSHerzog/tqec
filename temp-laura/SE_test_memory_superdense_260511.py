from tqec.computation.prism import BasisPrism, Position3DHex, ZXPrism
from tqec.computation.pipe_prism import PrismPipeKind
from tqec.computation.prism_graph import PrismGraph
from tqec.computation.correlation import find_correlation_surfaces

import tqec.computation.syndrome_extraction_cc as se


import temp_utils

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sinter

from datetime import datetime


g = PrismGraph("mem")

prisms = [
    (Position3DHex(0,0,-1), "ZZ", "init"),
]

for pos, kind, label in prisms:
    g.add_prism(pos, kind, label)


timestamp = datetime.now().strftime('%y%m%d_%H%M%S')


if __name__ == '__main__':

    d_lst = [3,5]
    circuit_builders = [se.SyndromeExtractionStimCC(g, d=d) for d in d_lst]

    p_values = list(np.round(np.geomspace(0.001, 0.05, 8), 8)) #originally 0.0001

    rounds = d_lst

    num_workers = 4

    se_type = "run_all_superdense"

    add_missing_detectors = False

    cs_list = []

    path = f"mem_sinter_superdense_{timestamp}.csv"

    stats = se.run_experiment_sinter(
        circuit_builders=circuit_builders,
        rounds=rounds,
        p_values=p_values,
        num_workers=num_workers,
        path = path,
        add_missing_detectors = add_missing_detectors,
        se_type = se_type,
        cs_list = cs_list
    )

    fig = se.plot_experiment_sinter(stats, d_lst)
    fig.savefig(f"LER_mem_superdense_{timestamp}.pdf")


