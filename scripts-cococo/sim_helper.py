# cnot_simulation_helper.py

from pathlib import Path

import matplotlib.pyplot as plt
import sinter

import tqec.computation.syndrome_extraction_cc as se
from tqec import NoiseModel
from tqec.computation.correlation import CorrelationSurface
from tqec.computation.prism_graph import PrismGraph
from tqec.simulation.plotting.inset import plot_observable_as_inset
from tqec.simulation.simulation import start_simulation_using_sinter
from tqec.simulation.split import split_stats_for_observables


def run_and_plot(
    *,
    # graphs and observables
    g_cc: PrismGraph,
    g_sc,
    cs_cc_lst: list[CorrelationSurface],
    cs_sc_lst: list[CorrelationSurface],
    zx_cc,
    # simulation parameters
    d_lst: list[int],
    p_values: list[float],
    num_workers: int,
    max_shots: int,
    max_errors: int,
    # paths
    tele_type: str,
    timestamp: str,
    # optional
    correlated_decoder,
    manhattan_radius: int = 2,
    add_missing_detectors: bool = False,
    run_simulation: bool = True,
):
    """Run and plot simulations."""
    assert len(cs_sc_lst) == len(cs_cc_lst), "same operators have to be applied on both codes."
    n_obs = len(cs_cc_lst)
    ks = [(d - 1) / 2 for d in d_lst]

    path_cc = (
        f"cnot_sinter_{tele_type}_{timestamp}_maxshots{max_shots}_maxerrors{max_errors}_cc.csv"
    )
    path_sc = (
        f"cnot_sinter_{tele_type}_{timestamp}_maxshots{max_shots}_maxerrors{max_errors}_sc_correlated.csv"
    )
    database_path_sc = (
        f"cnot_sinter_{tele_type}_{timestamp}_maxshots{max_shots}_maxerrors{max_errors}_sc_correlated.pkl"
    )

    if run_simulation:
        stats_sc = start_simulation_using_sinter(
            g_sc,
            ks=ks,
            ps=p_values,
            noise_model_factory=NoiseModel.uniform_depolarizing,
            manhattan_radius=manhattan_radius,
            observables=cs_sc_lst,
            decoders=["correlated_pymatching"],
            num_workers=num_workers,
            max_shots=max_shots,
            max_errors=max_errors,
            save_resume_filepath=Path(path_sc),
            database_path=Path(database_path_sc),
            custom_decoders={"correlated_pymatching": correlated_decoder},
            split_observable_stats=True,
        )

        print("SC FINISHED, STARTING CC...")

        circuit_builders = [se.SyndromeExtractionStimCC(g_cc, d=d) for d in d_lst]
        stats_cc = se.run_experiment_sinter(
            circuit_builders=circuit_builders,
            rounds=d_lst,
            p_values=p_values,
            num_workers=num_workers,
            max_shots=max_shots,
            max_errors=max_errors,
            path=path_cc,
            add_missing_detectors=add_missing_detectors,
            cs_lst=cs_cc_lst,
        )

    # reload from CSV (always, so plots are consistent whether we just ran or are reloading)
    stats_cc = sinter.read_stats_from_csv_files(path_cc)
    stats_sc = sinter.read_stats_from_csv_files(path_sc)

    for s in stats_sc:
        print(s.json_metadata, "shots:", s.shots, "errors:", s.errors, "discards:", s.discards)

    split_cc = split_stats_for_observables(stats_cc, num_observables=n_obs)
    split_sc = split_stats_for_observables(stats_sc, num_observables=n_obs)

    for obs_idx in range(n_obs):
        fig, ax = plt.subplots()

        sinter.plot_error_rate(
            ax=ax,
            stats=split_sc[obs_idx],
            x_func=lambda s: s.json_metadata["p"],
            group_func=lambda s: f"Surface Code d={int(s.json_metadata['d'])}",
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
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))

        fig.savefig(
            f"cnot_error_rate_{tele_type}_{timestamp}_obs{obs_idx}_maxerrors{max_errors}_maxshots{max_shots}.pdf"
        )
        plt.close(fig)
