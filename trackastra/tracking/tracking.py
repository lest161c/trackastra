import logging
from collections.abc import Callable, Hashable
from itertools import chain

import networkx as nx
import numpy as np
import scipy
from tqdm import tqdm

from .track_graph import TrackGraph

# from trackastra.tracking import graph_to_napari_tracks, graph_to_ctc

logger = logging.getLogger(__name__)


def copy_edge(edge: tuple, source: nx.DiGraph, target: nx.DiGraph):
    if edge[0] not in target.nodes:
        target.add_node(edge[0], **source.nodes[edge[0]])
    if edge[1] not in target.nodes:
        target.add_node(edge[1], **source.nodes[edge[1]])
    target.add_edge(edge[0], edge[1], **source.edges[(edge[0], edge[1])])


def track_greedy(
    candidate_graph,
    allow_divisions=True,
    threshold=0.5,
    edge_attr="weight",
):
    """Greedy matching, global.

    Iterates over global edges sorted by weight, and keeps edge if feasible and weight above threshold.

    Args:
       allow_divisions (bool, optional):
            Whether to model divisions. Defaults to True.

    Returns:
        solution_graph: NetworkX graph of tracks
    """
    logger.info("Running greedy tracker")

    solution_graph = nx.DiGraph()

    # TODO bring back
    # if args.gt_as_dets:
    # solution_graph.add_nodes_from(candidate_graph.nodes(data=True))

    edges = candidate_graph.edges(data=True)
    edges = sorted(
        edges,
        key=lambda edge: edge[2][edge_attr],
        reverse=True,
    )

    for edge in tqdm(edges, desc="Greedily matched edges"):
        node_in, node_out, features = edge
        assert features[edge_attr] <= 1.0, (
            "Edge weights are assumed to be normalized to [0,1]"
        )
        # assumes sorted edges
        if features[edge_attr] < threshold:
            break
        # Check whether this edge is a feasible edge to add
        # i.e. no fusing
        if node_out in solution_graph.nodes and solution_graph.in_degree(node_out) > 0:
            # target node already has an incoming edge
            continue
        if node_in in solution_graph and solution_graph.out_degree(node_in) >= (
            2 if allow_divisions else 1
        ):
            # parent node already has max number of outgoing edges
            continue
        # otherwise add to solution
        copy_edge(edge, candidate_graph, solution_graph)

    # df, masks = graph_to_ctc(solution_graph, masks_original)
    # tracks, tracks_graph, _ = graph_to_napari_tracks(solution_graph)

    return solution_graph
    # TODO this should all be in a tracker class
    # return df, masks, solution_graph, tracks_graph, tracks, candidate_graph


def build_graph(
    nodes: dict,
    weights: tuple | None = None,
    use_distance: bool = False,
    max_distance: int | None = None,
    max_neighbors: int | None = None,
    delta_t=1,
    weight_lookup: Callable[[Hashable, Hashable], float | None] | None = None,
) -> nx.DiGraph:
    """Build a candidate tracking graph from per-node detections.

    Candidate edges connect nodes in frames ``t`` and ``t + delta_t`` that
    are within ``max_distance`` and among the ``max_neighbors`` nearest
    neighbours. Edge weights come from one of three sources, in this
    precedence order:

    1. ``use_distance=True`` ignores ``weights`` and ``weight_lookup`` and
       assigns ``1 - distance / max_distance``.
    2. A non-empty ``weights`` iterable of ``((node_in, node_out), weight)``
       pairs assigns the supplied weight; pairs absent from the iterable are
       skipped.
    3. When ``weights`` is ``None`` and ``use_distance`` is ``False``, an
       optional ``weight_lookup`` callable is asked for each candidate edge.
       A non-``None`` return value adds the edge with that weight; a
       ``None`` return value skips it. This lazy path avoids materialising
       the full ``N x N`` weight dictionary.

    Args:
        nodes: Iterable of node dictionaries with ``id``, ``time``,
            ``label``, and ``coords`` entries.
        weights: Optional iterable of ``((node_in, node_out), weight)``
            pairs. Takes precedence over ``weight_lookup``.
        use_distance: If ``True``, use distance-derived weights and ignore
            both ``weights`` and ``weight_lookup``.
        max_distance: Maximum spatial distance for a candidate edge, or
            ``None`` for no limit.
        max_neighbors: Maximum number of accepted candidate edges per source
            node, or ``None`` for no limit.
        delta_t: Maximum frame gap between connected nodes.
        weight_lookup: Optional read-only callable mapping
            ``(node_in, node_out)`` to a weight or ``None``. Only consulted
            when ``weights`` is ``None`` and ``use_distance`` is ``False``.

    Returns:
        Directed candidate graph whose edges point forward in time.
    """
    logger.info(f"Build candidate graph with {delta_t=}")
    G = nx.DiGraph()

    if len(nodes) == 0:
        logger.warning("No nodes provided, returning empty graph")
        return G

    for node in nodes:
        G.add_node(
            node["id"],
            time=node["time"],
            label=node["label"],
            coords=node["coords"],
            # index=node["index"],
            weight=1,
        )

    use_weight_lookup = (
        weight_lookup is not None and weights is None and not use_distance
    )

    if use_distance:
        weights = None
    if weights is not None:
        weights = {w[0]: w[1] for w in weights}

    graph = TrackGraph(G, frame_attribute="time")
    frame_pairs = zip(
        chain(*[
            list(range(graph.t_begin, graph.t_end - d)) for d in range(1, delta_t + 1)
        ]),
        chain(*[
            list(range(graph.t_begin + d, graph.t_end)) for d in range(1, delta_t + 1)
        ]),
    )
    iterator = tqdm(
        frame_pairs,
        total=(graph.t_end - graph.t_begin) * delta_t,
        leave=False,
    )
    for t_begin, t_end in iterator:
        n_edges_t = len(G.edges)
        ni, nj = graph.nodes_by_frame(t_begin), graph.nodes_by_frame(t_end)

        if len(ni) == 0:
            # skip edge creation for empty frames
            logger.warning(f"No nodes in frame {t_begin}")
            continue

        if len(nj) == 0:
            # skip edge creation for empty frames
            logger.warning(f"No nodes in frame {t_end}")
            continue

        pi = []
        for _ni in ni:
            pi.append(np.array(G.nodes[_ni]["coords"]))
        pi = np.stack(pi)
        pj = []
        for _nj in nj:
            pj.append(np.array(G.nodes[_nj]["coords"]))
        pj = np.stack(pj)
        nj_array = np.asarray(nj)

        dists = scipy.spatial.distance.cdist(pi, pj)

        def add_candidate(_ni, _j, dist_row):
            dist = dist_row[_j]
            if max_distance is not None and dist > max_distance:
                return False
            if use_weight_lookup:
                lookup_weight = weight_lookup(_ni, nj_array[_j])
                if lookup_weight is None:
                    return False
                G.add_edge(_ni, nj_array[_j], weight=lookup_weight)
            elif weights is None:
                G.add_edge(
                    _ni, nj_array[_j], weight=1 - dist / max_distance
                )
            else:
                edge_weight = weights.get((_ni, nj_array[_j]))
                if edge_weight is None:
                    return False
                G.add_edge(_ni, nj_array[_j], weight=edge_weight)
            return True

        for _i, _ni in enumerate(ni):
            dist_row = dists[_i]
            if max_neighbors and len(nj) > max_neighbors:
                candidate_count = min(len(nj), max_neighbors * 4)
                candidate_indices = np.argpartition(
                    dist_row, candidate_count - 1
                )[:candidate_count]
                candidate_indices = candidate_indices[
                    np.argsort(dist_row[candidate_indices])
                ]
            else:
                candidate_indices = np.argsort(dist_row)

            neighbors = 0
            for _j in candidate_indices:
                if max_neighbors and neighbors >= max_neighbors:
                    break
                if add_candidate(_ni, _j, dist_row):
                    neighbors += 1

            if (
                max_neighbors
                and neighbors < max_neighbors
                and len(candidate_indices) < len(nj)
            ):
                remaining_indices = np.setdiff1d(
                    np.arange(len(nj)),
                    candidate_indices,
                    assume_unique=True,
                )
                for _j in remaining_indices[
                    np.argsort(dist_row[remaining_indices])
                ]:
                    if neighbors >= max_neighbors:
                        break
                    if add_candidate(_ni, _j, dist_row):
                        neighbors += 1

        e_added = len(G.edges) - n_edges_t
        if e_added == 0:
            logger.warning(f"No candidate edges in frame {t_begin}")
        iterator.set_description(
            f"{e_added} edges in frame {t_begin}  Total edges: {len(G.edges)}"
        )

    logger.info(f"Added {len(G.nodes)} vertices, {len(G.edges)} edges")

    return G
