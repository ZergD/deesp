""" This file is the main file for the Expert Agent called Deesp """
import functools
import datetime
import math
import os

import pprint

import numpy as np
import networkx as nx
from graphviz import Source


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
        self.are_loads = None
        self.are_prods = None
        # custom layout for the graph to look like the simulator Pypownet.
        self.custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                              (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216)]
        # dictionnary containing all interesting electric paths: constrained path, parallel path, loop path
        self.local_epaths = None
        self.lines_por_values = None

    def load(self, _grid):
        """ This function loads the data representing the current state of the network
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
        self.id_line_cut = 30
        if self.debug:
            print("============================= FUNCTION compute_load_outage =============================")
            print("line statuses = ", self.grid.get_lines_status())

        # self.grid.get_lines_status()[self.id_line_cut] = 0

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
            print("we recompute for test!!!!")
            self.new_e_flows = self.grid.extract_flows_a()
            print("new new e flows = ", self.new_e_flows)

    def build_nodes(self, custom_layout):
        # =========================================== NODE PART ===========================================
        i = 0
        # We color the nodes depending if they are production or consumption
        for value, is_prod, is_load in zip(custom_layout, self.are_prods, self.are_loads):
            if is_prod:
                self.g.add_node(i + 1, pos=(str(value[0]) + ", " + str(value[1]) + "!"), pin=True,
                                prod_or_load="prod", style="filled", fillcolor="#f30000")  # red color
            else:
                self.g.add_node(i + 1, pos=(str(value[0]) + ", " + str(value[1]) + "!"), pin=True,
                                prod_or_load="load", style="filled", fillcolor="#478fd0")  # blue color
            i += 1

    def build_edges(self, custom_layout, gtype="powerflow"):
        if self.debug:
            ar = list(zip(self.idx_or, self.idx_ex, self.delta_e_flows, self.initial_e_flows))
            print("ZIP OF DEATH = ")
            pprint.pprint(ar)

        if gtype is "powerflow":
            for origin, extremity, reported_flow, current_flow, line_por in zip(self.idx_or, self.idx_ex,
                                                                                self.delta_e_flows,
                                                                                self.initial_e_flows,
                                                                                self.lines_por_values):
                origin += 1
                extremity += 1
                penwidth = math.fabs(current_flow) / 5
                if penwidth == 0.0:
                    penwidth = 0.1

                if line_por >= 0:
                    self.g.add_edge(extremity, origin, xlabel="%.2f" % current_flow, color="gray", fontsize=10,
                                    penwidth="%.2f" % penwidth)
                else:
                    self.g.add_edge(origin, extremity, xlabel="%.2f" % current_flow, color="gray", fontsize=10,
                                    penwidth="%.2f" % penwidth)

        elif gtype is "overflow":
            i = 0
            for origin, extremity, reported_flow, current_flow, line_por in zip(self.idx_or,
                                                                                self.idx_ex, self.delta_e_flows,
                                                                                self.initial_e_flows,
                                                                                self.lines_por_values):
                origin += 1
                extremity += 1
                penwidth = math.fabs(reported_flow) / 5
                if penwidth == 0.0:
                    penwidth = 0.1
                if i == self.id_line_cut:
                    self.g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="black",
                                    style="dotted, setlinewidth(2)", fontsize=10, penwidth="%.2f" % penwidth,
                                    constrained=True)
                elif reported_flow < 0:
                    if line_por >= 0:
                        self.g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                    else:
                        self.g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                else:  # > 0
                    if line_por >= 0:
                        self.g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                    else:
                        self.g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                i += 1
        else:
            raise RuntimeError("GType of graph to build not understood!")

    def build_graph(self, axially_symetric=False, gtype="powerflow"):

        # We start building the Graph with NetworkX here.
        # =========================================== GRAPH PART ===========================================
        self.g = nx.DiGraph()

        custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54), (-64, 54), (366, 0),
                              (438, 0), (326, 54), (222, 108), (79, 162), (-152, 270), (-64, 270), (222, 216)]
        if axially_symetric:
            x_inversed_layout = []
            for x in custom_layout:
                x_inversed_layout.append((x[0] * -1, x[1]))
            custom_layout = x_inversed_layout

        # =========================================== NODE PART ===========================================
        self.build_nodes(custom_layout)
        # =========================================== EDGE PART ===========================================
        self.build_edges(custom_layout, gtype)

        # self.save_to_file("")

    def build_overload_graph(self, axially_symetric=False):
        """We build an overload graph.
        First we check if we computed the load outage"""

        if self.compute_load_outage.has_been_called and self.retrieve_topology.has_been_called:
            if self.debug:
                print("============================= FUNCTION build_overload_graph =============================")
                print("Functions: compute_load_outage() and retrieve_topology have both been called, we can build the "
                      "overload_graph")

            if axially_symetric:
                x_inversed_layout = []
                for x in self.custom_layout:
                    x_inversed_layout.append((x[0] * -1, x[1]))
                self.custom_layout = x_inversed_layout

            # We start building the Graph with NetworkX here.
            # =========================================== GRAPH PART ===========================================
            self.g = nx.DiGraph()
            # =========================================== NODE PART ===========================================
            i = 0
            # We color the nodes depending if they are production or consumption
            for value, is_prod, is_load in zip(self.custom_layout, self.are_prods, self.are_loads):
                if is_prod:
                    self.g.add_node(i + 1, pos=(str(value[0]) + ", " + str(value[1]) + "!"), pin=True,
                                    prod_or_load="prod", style="filled", fillcolor="#f30000")  # red color
                else:
                    self.g.add_node(i + 1, pos=(str(value[0]) + ", " + str(value[1]) + "!"), pin=True,
                                    prod_or_load="load", style="filled", fillcolor="#478fd0")  # blue color
                i += 1
            # =========================================== EDGE PART ===========================================
            i = 0
            if self.debug:
                ar = list(zip(self.idx_or, self.idx_ex, self.delta_e_flows, self.initial_e_flows))
                print("ZIP OF DEATH = ")
                pprint.pprint(ar)
            for origin, extremity, reported_flow, current_flow, line_por in zip(self.idx_or, self.idx_ex,
                                                                                self.delta_e_flows,
                                                                                self.initial_e_flows,
                                                                                self.lines_por_values):
                origin += 1
                extremity += 1
                penwidth = math.fabs(reported_flow) / 5
                if penwidth == 0.0:
                    penwidth = 0.1

                if i == self.id_line_cut:
                    self.g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="black",
                                    style="dotted, setlinewidth(2)", fontsize=10, penwidth="%.2f" % penwidth,
                                    constrained=True)
                elif reported_flow < 0:
                    if line_por >= 0:
                        self.g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                    else:
                        self.g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                else:  # > 0
                    if line_por >= 0:
                        self.g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                    else:
                        self.g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                                        penwidth="%.2f" % penwidth)
                i += 1

            if self.debug:
                print("============================= FINISHED building overload graph =============================")

        # Else, the functions build_overload_graph and retrieve_topology have not been run
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

        # retrieve loads and prods
        nodes_ids = mpcbus[:, 0]
        prods_ids = mpcgen[:, 0]
        self.are_prods = np.logical_or([node_id in prods_ids for node_id in nodes_ids[:len(nodes_ids) // 2]],
                                  [node_id in prods_ids for node_id in nodes_ids[len(nodes_ids) // 2:]])
        self.are_loads = np.logical_or(self.grid.are_loads[:len(mpcbus) // 2], self.grid.are_loads[len(nodes_ids) // 2:])
        self.lines_por_values = self.grid.mpc['branch'][:, 13]

        if self.debug:
            print("============================= FUNCTION retrieve_topology =============================")
            print("self.idx_or = ", self.idx_or)
            print("self.idx_ex = ", self.idx_ex)
            print("self.lines_por_values = ", self.lines_por_values)
            print("Nodes that are prods =", self.are_prods)
            print("Nodes that are loads =", self.are_loads)

    def display_graph(self, display_type: str):
        """ This function displays a graph
        :param display_type: either "geo" or "elec"
        :param axially_symetric: if True, it performs an axial symetry on the graph, (only for display purpose)
        :return:
        """

        assert(isinstance(display_type, str))

        # we create filenames
        folder_output = "./deesp/ressources/output/"
        # current_date_no_filter = datetime.datetime.now()
        # current_date = current_date_no_filter.strftime("%Y-%m-%d_%H-%M")
        # filename_dot = "graph_result_" + display_type + "_" + current_date + ".dot"
        # filename_pdf = "graph_result_" + display_type + "_" + current_date + ".pdf"
        filename_dot = "graph_result_" + display_type + ".dot"
        filename_pdf = "graph_result_" + display_type + ".pdf"
        hard_filename_dot = folder_output + filename_dot
        # hard_filename_dot = filename_dot
        hard_filename_pdf = folder_output + filename_pdf
        if self.debug:
            print("============================= FUNCTION display_graph =============================")
            print("hard_filename = ", hard_filename_pdf)

        # we save the graph in a dot file
        nx.drawing.nx_pydot.write_dot(self.g, hard_filename_dot)

        if display_type is "geo":
            cmd_line = "neato -n -Tpdf " + hard_filename_dot + " -o " + hard_filename_pdf
            if self.debug:
                print("we print the cmd line = ", cmd_line)
            os.system(cmd_line)
            os.system("evince " + hard_filename_pdf + " &")

        elif display_type is "elec":
            layout_engines = ["dot"]
            for layout in layout_engines:
                # this line reads from original_filename
                gg = Source.from_file(hard_filename_dot, engine=layout)
                # this line creates files
                # the view function adds .pdf to the filename so we remove it first

                filename, file_extention = os.path.splitext(hard_filename_pdf)
                gg.view(filename=filename, cleanup=True)

    def compute_meaningful_structures(self):
        """Computes meaningful structures from the main graph: constrained paths, //paths, up/down-stream areas,
         etc..."""
        self.dev_graph_courses_tests()
        self.local_epaths = self.get_local_epaths()

    def get_local_epaths(self) -> list:
        """This function returns a dictionnary of all interesting paths:
        Constrained path, parallel path, loop path,
        """
        # get constrained_paths
        self.get_constrained_paths()
        return []

    def get_constrained_paths(self):
        """Retrieves the constrained paths. A small graph representing the constrained path"""

        assert(isinstance(self.g, nx.DiGraph))

        # array containing the indices of edges with positive report flow
        pos_edges = []
        # get indices of positive edges
        i = 1
        for u, v, report in self.g.edges(data="xlabel"):
            if float(report) > 0:
                pos_edges.append((i, (u, v)))
            i += 1

        print(f"pos_edges = {pos_edges}")

        # we first remove the constrained edge - "black one"

        # TODO TO CONTINUE
        # delete from graph positive edges
        # this extracts the (u,v) from pos_edges
        # print("pos_edges test = ", list(zip(*pos_edges))[1])
        # self.g.remove_edges_from(list(zip(*pos_edges))[1])

        # self.display_graph("geo")

    def dev_graph_courses_tests(self):
        if self.debug:
            for n, nbrsdict in self.g.adjacency():
                print(f"n = {n}, nbrsdict = {nbrsdict}")

            for u, v, report in self.g.edges(data="xlabel"):
                print(f"u = {u}, v = {v}, report = {report}")

