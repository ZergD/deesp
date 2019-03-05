""" This file is the main file for the Expert Agent called Deesp """


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
        self.initial_e_flows = None
        self.new_e_flows = None
        self.delta_e_flows = None

    def load(self, _grid):
        """
        This function loads the data representing the current state of the network
        :param _grid:
        :return:
        """
        self.grid = _grid

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

    # def build_overlord_graph(self):
    #     self.compute_load_outage()
