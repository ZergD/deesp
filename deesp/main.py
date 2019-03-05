__author__ = "MarcM"

import argparse
from deesp.core import Deesp

parser = argparse.ArgumentParser(description="Expert System")
parser.add_argument("-d", "--debug", default=False, type=bool,
                    help="Prints additional information for debugging purposes")


def main():
    print("Deesp Agent created...")
    deesp = Deesp()

    # we load the data from timestep t, usually an Overloaded situation
    # deesp.load()

    # Compute Load Outage Distribution Factor of overloaded lines
    # deesp.compute_load_outage()

    # Build Overload Distribution Graph
    # deesp.build_overload_graph()

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
