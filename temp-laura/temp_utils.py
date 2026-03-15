import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mqt.qecc
import numpy as np
from matplotlib.patches import Polygon

from tqec.computation.prism import Position3DHex


def build_position_encoding(
    stabilizers: list[list[Position3DHex]],
) -> dict[Position3DHex, int]:
    """Create a bijective mapping from Position3DHex to integers."""
    all_positions = {pos for stab in stabilizers for pos in stab}
    return {pos: idx for idx, pos in enumerate(all_positions)}


def translate_stabilizers(
    stabilizers: list[list[Position3DHex]],
    pos_to_int: dict[Position3DHex, int],
) -> list[list[int]]:
    """Translate stabilizers from Position3DHex to integer encoding."""
    return [[pos_to_int[pos] for pos in stab] for stab in stabilizers]

def build_check_matrix(
    encoded_stabilizers: list[list[int]],
    n_positions: int,
) -> list[list[int]]:
    """Build a check matrix from encoded stabilizers.

    Rows correspond to stabilizers, columns to positions.
    Entry (i, j) is 1 if position j appears in stabilizer i.
    """
    matrix = [[0] * n_positions for _ in range(len(encoded_stabilizers))]
    for row, stab in enumerate(encoded_stabilizers):
        for pos_int in stab:
            matrix[row][pos_int] = 1
    return np.array(matrix)

def check_logical_operator(xl, hz):
    """Check whether given logical operator commutes with given stabilizers.

    Important: for a Z logical operator, check the X stabilizers and vice versa. 
    """
    values = []
    for i in range(np.shape(hz)[0]):
        stab = hz[i,:]
        res = (stab @ xl[0])%2 
        values.append(res)
    if np.sum(values) == 0.0:
        return True
    else:
        return False


def plot_position_dict(
    *,
    size,
    bdry: dict[str, list] | None = None,
    stabilizers: list[list] | None = None,
    star_op: list[list] | None = None,
) -> plt.Figure:
    """Plot Position3DHex positions using their to_euclidean coords.

    Args:
        bdry:        dict[str, list[Position3DHex]] — each key is a label, plotted as scattered dots.
        stabilizers: list[list[Position3DHex]] — each inner list plotted with its own color,
                     dots at each position and a filled polygon connecting them.
        star_op:     list[Position3DHex] — plotted as black crosses.
    """
    fig, ax = plt.subplots(figsize=size)
    n_bdry        = len(bdry)        if bdry        is not None else 0
    n_stabilizers = len(stabilizers) if stabilizers is not None else 0
    n_total = n_bdry + n_stabilizers

    def rainbow(i: int) -> tuple:
        return cm.rainbow(i / max(n_total - 1, 1))

    color_idx = 0
    if stabilizers is not None:
        for positions in stabilizers:
            if not positions:
                continue
            if len(positions) == 2:
                color = "black"
            else:
                color = rainbow(color_idx); color_idx += 1
            xs = [p.to_euclidean()[0] for p in positions]
            ys = [p.to_euclidean()[1] for p in positions]
            poly = Polygon(list(zip(xs, ys)), closed=True,
                        facecolor=(*color[:3], 0.5) if color != "black" else (0, 0, 0, 0.2),
                        edgecolor=color, linewidth=1.5, zorder=2)
            ax.add_patch(poly)
            ax.scatter(xs, ys, color=color, s=60, zorder=3)
            for p, x, y in zip(positions, xs, ys):
                ax.text(x, y, f"({p.x},{p.y})", fontsize=7)

    if bdry is not None:
        for label, positions in bdry.items():
            color = rainbow(color_idx); color_idx += 1
            xs = [p.to_euclidean()[0] for p in positions]
            ys = [p.to_euclidean()[1] for p in positions]
            ax.scatter(xs, ys, color=color, label=label, s=80, zorder=3)
            for p, x, y in zip(positions, xs, ys):
                ax.annotate(f"({p.x},{p.y})", (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)

    if star_op is not None:
        colors = plt.cm.tab10.colors  # or any colormap with enough distinct colors
        for k, op in enumerate(star_op):
            xs = [p.to_euclidean()[0] for p in op]
            ys = [p.to_euclidean()[1] for p in op]
            color = colors[k % len(colors)]
            # Colored background circle
            ax.scatter(xs, ys, color=color, s=200, zorder=3, alpha=0.6, label=f"Star Operator {k}")
            # Black X on top
            ax.scatter(xs, ys, color="black", marker="x", s=80, linewidths=1.5, zorder=4)

    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Position3DHex boundary vertices")
    plt.tight_layout()
