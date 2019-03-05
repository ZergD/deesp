""" This file is the main file for the Expert Agent called Deesp """
import functools
import numpy as np


def calltracker(func):
    """A wrapper to check if a function has already been called or not """
    @functools.wraps(func)
    def wrapper(*args):
        wrapper.has_been_called = True
        return func(*args)
    wrapper.has_been_called = False
    return wrapper


class Deesp:
    """ Summary of class here. """
    def __init__(self, debug=False):
        """Instanciates a Dispatcher Deesp"""
        print("Deesp init executed ...")
        self.debug = debug
        self.param_path = None

        # this is the main graph representing the network
        self.g = None
        # this is the grid from pypownet
        self.grid = None
        self.id_line_cut = None
        # electric flows, array representing the flows on each edge
        self.initial_e_flows = None
        self.new_e_flows = None
        self.delta_e_flows = None
        # topology part, idx_or is an array of edges representing the edge's origins. Same respectively with idx_ex.
        self.idx_or = None
        self.idx_ex = None

    def load(self, _grid):
        """
        This function loads the data representing the current state of the network
        :param _grid:
        :return:
        """
        self.grid = _grid

    @calltracker
    def compute_load_outage(self):
        """La perte de charge, Compute Load Outage Distribution Factor of overloaded lines"""
        assert(self.grid is not None)

        depth = 0
        fname_end = '_cascading%d' % depth

        # we compute initial_e_flows
        self.initial_e_flows = self.grid.extract_flows_a()

        # we disable the line 10
        self.id_line_cut = 10
        if self.debug:
            print("============================= FUNCTION compute_load_outage =============================")
            print("line statuses = ", self.grid.get_lines_status())

        self.grid.get_lines_status()[self.id_line_cut] = 0

        if self.debug:
            print("line statuses = ", self.grid.get_lines_status())

        # now that a line has been cut, we recompute the flows and extract them
        self.grid.compute_loadflow(fname_end)
        self.new_e_flows = self.grid.extract_flows_a()

        # we compute the delta between initial and new
        self.delta_e_flows = self.new_e_flows - self.initial_e_flows
        if self.debug:
            print("initial_e_flows = ", self.initial_e_flows)
            print("new_e_flows = ", self.new_e_flows)
            print("delta_e_flows = ", self.delta_e_flows)

    def build_overload_graph(self):
        """We build an overload graph.
        First we check if we computed the load outage"""

        if self.compute_load_outage.has_been_called and self.retrieve_topology.has_been_called:
            if self.debug:
                print("============================= FUNCTION build_overload_graph =============================")
                print("Functions: compute_load_outage() and retrieve_topology have both been called, we can build the "
                      "overload_graph")
                pass
        else:
            if self.debug:
                raise RuntimeError("Error, function \"{}\" or \"{}\" has not been called yet".format(
                    self.build_overload_graph.__name__, self.retrieve_topology.__name__))

    @calltracker
    def retrieve_topology(self):
        """This function retrieves the topology"""
        # retrieve topology
        mpcbus = self.grid.mpc['bus']
        mpcgen = self.grid.mpc['gen']
        half_nodes_ids = mpcbus[:len(mpcbus) // 2, 0]
        node_to_substation = lambda x: int(float(str(x).replace('666', '')))
        # intermediate step to get idx_or and idx_ex
        nodes_or_ids = np.asarray(list(map(node_to_substation, self.grid.mpc['branch'][:, 0])))
        nodes_ex_ids = np.asarray(list(map(node_to_substation, self.grid.mpc['branch'][:, 1])))
        # origin
        self.idx_or = [np.where(half_nodes_ids == or_id)[0][0] for or_id in nodes_or_ids]
        # extremeties
        self.idx_ex = [np.where(half_nodes_ids == ex_id)[0][0] for ex_id in nodes_ex_ids]
        if self.debug:
            print("============================= FUNCTION retrieve_topology =============================")
            print("self.idx_or = ", self.idx_or)
            print("self.idx_ex = ", self.idx_ex)






