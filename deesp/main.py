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
    parameters_folder = "/home/mozgawamar/Documents/deesp/deesp/ressources/parameters/default14"
    game_level = "level0"
    chronic_looping_mode = 'natural'
    chronic_starting_id = 0
    game_over_mode = 'soft'

    # load a game
    _game = ggame.Game(parameters_folder, game_level, chronic_looping_mode, chronic_starting_id, game_over_mode,
                       renderer_frame_latency=20)

    # retrieve a grid
    _grid = _game.grid

    deesp = Deesp(args.debug)
    print("Deesp Agent created...")

    # we load the data from timestep t, usually an Overloaded situation
    deesp.load(_grid)

    # La perte de charge, Compute Load Outage Distribution Factor of overloaded lines
    deesp.compute_load_outage()

    # we retrieve the topology
    deesp.retrieve_topology()

    # Build Overload Distribution Graph
    deesp.build_overload_graph()

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
