import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
    mapping: dict | None = None,
    mapping_ancillas: dict[tuple, int] | None = None
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

    ancilla_lookup = None
    if mapping_ancillas is not None:
        ancilla_lookup = {
            frozenset(k): v for k, v in mapping_ancillas.items()
        }

    def draw_mapping_label(ax, p, x, y):
        label = f"({p.x},{p.y})"
        p = Position3DHex(p.x, p.y, 0)
        if mapping is not None and p in mapping:
            label += f"\n{mapping[p]}"   # <-- add value on new line

        ax.text(
            x, y,
            label,
            fontsize=7,
            ha="center",
            va="center",
            zorder=5,
            #bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1)
        )

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
                if mapping is None:
                    ax.text(x, y, f"({p.x},{p.y})", fontsize=7)
                else:
                    draw_mapping_label(ax, p, x, y)

            if ancilla_lookup is not None:
                positions_z = tuple(Position3DHex(pos.x, pos.y, 0) for pos in positions)
                key = frozenset(positions_z)
                if key in ancilla_lookup:
                    val = ancilla_lookup[key]

                    # compute centroid of plaquette
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)

                    ax.text(
                        cx, cy,
                        f"{val}",
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        color="black",
                        zorder=6,
                        #bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", pad=1)
                    )

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

def plot_position_dict_3color(
    *,
    size,
    color_assignment: dict[tuple, str],
    stabilizer_product: list[list[Position3DHex]] | None = None,
    star_operators=None,
    show_positions: bool = False,
) -> plt.Figure:
    """Plot Position3DHex positions using their to_euclidean coords."""
    fig, ax = plt.subplots(figsize=size)
    for positions, color in color_assignment.items():
        if not positions:
            continue
        pts = np.array([p.to_euclidean()[:2] for p in positions])
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        pts = pts[np.argsort(angles)]
        rgba = mcolors.to_rgba(color, alpha=0.5)
        poly = Polygon(pts, closed=True,
                       facecolor=rgba, edgecolor=color, linewidth=1.5, zorder=2)
        ax.add_patch(poly)
        if show_positions:
            for p in positions:
                pt = p.to_euclidean()[:2]
                ax.annotate(
                f"({p.x},{p.y})",
                xy=pt,
                fontsize=10,
                ha="center", va="center",
                zorder=8,
                color="grey",
            )
    if stabilizer_product is not None:
        for stab in stabilizer_product:
            if not stab:
                continue
            if len(stab) != 2:
                pts = np.array([p.to_euclidean()[:2] for p in stab])
                centroid = pts.mean(axis=0)
                angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
                pts = pts[np.argsort(angles)]
                poly = Polygon(pts, closed=True,
                            facecolor=(0, 0, 0, 0.3), edgecolor="black", linewidth=2.0, zorder=4)
                ax.add_patch(poly)
                ax.plot(
                    [centroid[0] - 0.3, centroid[0] + 0.3],
                    [centroid[1], centroid[1]],
                    color="black", linewidth=1.5, zorder=5
                )
                ax.plot(
                    [centroid[0], centroid[0]],
                    [centroid[1] - 0.3, centroid[1] + 0.3],
                    color="black", linewidth=1.5, zorder=5
                )
                continue
            pts = np.array([p.to_euclidean()[:2] for p in stab])
            ax.plot(pts[:, 0], pts[:, 1], color="dimgray", linewidth=3.0, zorder=6)
    if star_operators is not None:
        for star_operator in star_operators:
            for pos in star_operator:
                pt = pos.to_euclidean()[:2]
                ax.plot(
                    pt[0], pt[1],
                    marker="o", markersize=8,
                    color="deeppink", markeredgecolor="hotpink",
                    linewidth=0, zorder=7,
                )
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Stabilizers")
    plt.tight_layout()


#----------plotting for rectangular coords------------
def sort_by_angle(points):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

def plot_mapping_full(mapping_full, z, size = (12,8)):
    """Plot the mapping on the rectangular grid."""
    fig, ax = plt.subplots(figsize=size)

    all_stab_infos = []
    for prism_data in mapping_full[z].values():
        if prism_data.stabilizers:
            all_stab_infos.extend(prism_data.stabilizers)

    n_stabs = len(all_stab_infos)

    def rainbow(i):
        return cm.rainbow(i / max(n_stabs - 1, 1))

    # plot stabilizer faces
    for i, stab_info in enumerate(all_stab_infos):
        color = rainbow(i)
        coords = [(pe.rect[0], pe.rect[1]) for pe in stab_info.data_qubits]
        coords = sort_by_angle(coords)
        xs, ys = zip(*coords)
        poly = Polygon(list(zip(xs, ys)), closed=True,
                       facecolor=(*color[:3], 0.3),
                       edgecolor=color, linewidth=1.5, zorder=2)
        ax.add_patch(poly)


    # plot data qubits
    for prism_data in mapping_full[z].values():
        for pe in prism_data.positions:
            rx, ry = pe.rect
            ax.scatter(rx, ry, color='blue', s=80, zorder=3)
            txt = f"h({pe.hex.x},{pe.hex.y})\nr{pe.rect}\n#{pe.label}"
            ax.annotate(txt, (rx, ry), xytext=(4, 4), textcoords='offset points', fontsize=6, color='blue')

    # plot ancillas
    # plot ancillas
    for prism_data in mapping_full[z].values():
        if prism_data.stabilizers:
            for stab_info in prism_data.stabilizers:
                if stab_info.ancilla:
                    for anc in stab_info.ancilla:
                        rx, ry = anc.rect
                        ax.scatter(rx, ry, color='red', s=100, zorder=4, marker='*')
                        txt = f"r{anc.rect}\n#{anc.label}"
                        ax.annotate(txt, (rx, ry), xytext=(4, 4), textcoords='offset points', fontsize=6, color='red')

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"z={z}")
    plt.tight_layout()
    plt.show()