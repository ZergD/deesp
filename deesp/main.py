__author__ = "MarcM"

import argparse
from deesp.core import Deesp

import pypownet.environment
import pypownet.grid
import pypownet.game as ggame

parser = argparse.ArgumentParser(description="Expert System")
parser.add_argument("-d", "--debug", action="store_true",
                    help="Prints additional information for debugging purposes")


def main():
    args = parser.parse_args()
    # parameters_folder = "../pypownet-master/parameters/default14"
    parameters_folder = "./deesp/ressources/parameters/default14"
    game_level = "level0"
    chronic_looping_mode = 'natural'
    chronic_starting_id = 0
    game_over_mode = 'soft'

    # load a game
    _game = ggame.Game(parameters_folder, game_level, chronic_looping_mode, chronic_starting_id, game_over_mode,
                       renderer_frame_latency=20)
    # _game.render(None)

    # retrieve a grid
    _grid = _game.grid

    deesp = Deesp(args.debug)
    print("Deesp Agent created...")

    # we load the data from timestep t, usually an Overloaded situation
    deesp.load(_grid)

    g = deesp.build_graph(_grid, gtype="powerflow", axially_symetric=False)
    deesp.display_graph(g, "geo", name="powerflow_before_cut")

    # ######################## CUT AND RECOMPUTE #####################
    line_cut = 9
    depth = 0
    fname_end = '_cascading%d' % depth
    _grid.get_lines_status()[line_cut] = 0
    _grid.compute_loadflow(fname_end)
    # ######################## CUT AND RECOMPUTE #####################

    g = deesp.build_graph(_grid, gtype="powerflow", axially_symetric=False)
    deesp.display_graph(g, "geo", name="powerflow_after_cut")

    # La perte de charge, Compute Load Outage Distribution Factor of overloaded lines
    # deesp.compute_load_outage()  # compute the load outage of the line in overflow

    # we retrieve the topology
    # deesp.retrieve_topology()

    # Build Overload Distribution Graph
    # deesp.build_overload_graph()  # possible arguments in this function, change some vars
    # deesp.build_graph(axially_symetric=False)

    # Display the internal graph
    # deesp.display_graph("geo")  # powerflows => graph with powerflows, overload => overload graph

    # _game.render(None)

    # Computes meaningful structures from the main graph: constrained paths, //paths, up/down-stream areas, etc...
    # deesp.compute_meaningful_structures()

    # Identify local electric paths
    # deesp.get_local_epaths()

    # Identify substation type
    # deesp.get_substation_type()

    # Rank substations
    # deesp.rank_substations()

    # Rank topologies within substations

    # Compute load flow under selected topology

    # Score the selected topology


if __name__ == "__main__":
    main()
