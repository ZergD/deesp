# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # <font color=orange>1) Examples : Load a game with an initial grid state and get the flows in Amperes</font>

# ### <font color=green>I) Initialisation, game loading.</font>

# +
# %run ~/Documents/pypownet-master/pypownet/grid.py
# %run ~/Documents/pypownet-master/pypownet/game.py
from IPython.core.display import display, SVG

import networkx as nx

parameters_folder = "../../deesp/ressources/parameters/default14"
game_level = "level0"
chronic_looping_mode = 'natural'
chronic_starting_id = 0
game_over_mode = 'soft'

_game = Game(parameters_folder, game_level, chronic_looping_mode, chronic_starting_id,
                 game_over_mode, renderer_frame_latency=None)
# -

# ### <font color=green>II) We retrieve grid and currentflowsZZ</font>

_grid = _game.grid
initial_flows = _grid.extract_flows_a()
initial_flows

# # <font color=orange>2) Compute line flow redispatch for line 10

# ### <font color=green>I) We retrieve the lines statuses, then disable line 10 </font>

lineStatuses = _grid.get_lines_status()
lineStatuses

id_line_cut = 10
lineStatuses[id_line_cut] = 0
lineStatuses

# ### <font color=green>II) We recompute the load flows
# - a) Compute a load flow
# - b) Extract flows
# - c) Compute the difference between initial state and new situation</font>

# +
depth = 0 
fname_end ='_cascading%d' % depth

#recompute load flow
_grid.compute_loadflow(fname_end)

# extract flows
new_flows = _grid.extract_flows_a()
new_flows
# -

# diff between initial and new flows
delta_flows = new_flows - initial_flows
delta_flows

# # <font color=orange>3) Build the overloaded redispatch graph</font>
# 1) Get network topology
#
# 2) Build a graph for this topology
#
# 3) Give the graph's edges the weight of the flow report
#
# 4) Create a visualisation of the report graph

# retrieve topology
mpcbus = _grid.mpc['bus']
mpcgen = _game.grid.mpc['gen']
half_nodes_ids = mpcbus[:len(mpcbus) // 2, 0]
node_to_substation = lambda x: int(float(str(x).replace('666', '')))
# intermediate step to get idx_or and idx_ex
nodes_or_ids = np.asarray(list(map(node_to_substation, _game.grid.mpc['branch'][:, 0])))
nodes_ex_ids = np.asarray(list(map(node_to_substation, _game.grid.mpc['branch'][:, 1])))
# origin
idx_or = [np.where(half_nodes_ids == or_id)[0][0] for or_id in nodes_or_ids]
# extremeties
idx_ex = [np.where(half_nodes_ids == ex_id)[0][0] for ex_id in nodes_ex_ids]

# +
# create a Graph
g = nx.DiGraph()

# custom layout for it to look like the simulator Pypownet.
custom_layout = [(-280, -81), (-100, -270), (366, -270), (366, -54), (-64, -54),
                 (-64, 54), (366, 0), (438, 0), (326, 54), (222, 108), (79, 162),
                 (-152, 270), (-64, 270), (222, 216)]

# ================================================ NODE PART ===================================================
nodes_ids = mpcbus[:, 0]
prods_ids = mpcgen[:, 0]
are_prods = np.logical_or([node_id in prods_ids for node_id in nodes_ids[:len(nodes_ids) // 2]],
                          [node_id in prods_ids for node_id in nodes_ids[len(nodes_ids) // 2:]])
are_loads = np.logical_or(_grid.are_loads[:len(mpcbus) // 2],
                          _grid.are_loads[len(nodes_ids) // 2:])
print("Nodes that are prods =", are_prods)
print("Nodes that are loads =", are_loads)


i = 0
# We color the nodes
for value, is_prod, is_load in zip(custom_layout, are_prods, are_loads):
    if is_prod:
        g.add_node(i + 1, pos=(str(value[0]) + ", " + str(value[1]) + "!"), pin=True, prod_or_load="prod",
                   style="filled", fillcolor="#f30000")  # red color
    else:
        g.add_node(i + 1, pos=(str(value[0]) + ", " + str(value[1]) + "!"), pin=True, prod_or_load="load",
                   style="filled", fillcolor="#478fd0")  # blue color
    i += 1

# -

# ================================================ EDGE PART ===================================================
i = 1
for origin, extremity, reported_flow, current_flow in zip(idx_or, idx_ex, delta_flows, initial_flows):
    origin += 1
    extremity += 1
    penwidth = math.fabs(reported_flow) / 5
    if penwidth == 0.0:
        penwidth = 0.1

    if i == id_line_cut:
        g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="black",
                   style="dotted, setlinewidth(2)", fontsize=10, penwidth=penwidth)
    elif reported_flow < 0:
        if current_flow > 0:
            g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="blue", fontsize=10, 
                       penwidth=penwidth)
        else:
            g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="blue", fontsize=10,
                       penwidth=penwidth)

    else:  # > 0
        if current_flow > 0:
            g.add_edge(origin, extremity, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                       penwidth=penwidth)
        else:
            g.add_edge(extremity, origin, xlabel="%.2f" % reported_flow, color="red", fontsize=10,
                       penwidth=penwidth)

    # g.edges[origin, extremity]['id'] = i
    i += 1

# # REPRESENTATION ELECTRIQUE

# +
# display Graph
import graphviz

layout_engines = ["dot"]
original_filename = "./outputs/jupyter_test.dot"
final_filename = "./outputs/jupyter_res1.pdf"
nx.drawing.nx_pydot.write_dot(g, original_filename)
g
cmd_line = "neato -n -Tpdf " + original_filename + " -o " + final_filename
print("we print the cmd line = ", cmd_line)
os.system(cmd_line)
#os.system("evince " + final_filename + " &")

# this is the command line to create a pdf from a dot file, with fixed nodes
# neato - n2 - Tgif file.dot - o file.gif

for layout in layout_engines:
    filename = "./graph_results/pywpow_graph_" + layout + ".dot"
    gg = graphviz.Source.from_file(original_filename, engine=layout)
    ##gg.view(filename=filename)
gg
# -

# # REPRESENTATION GEOGRAPHIQUE

# +
from IPython.display import IFrame

IFrame("./outputs/jupyter_res1.pdf", width=800, height=700)
# -

# # <font color=orange>4) Identify Local Electrical Paths</font>

# Let us add here a final test
print("This is our final test")
