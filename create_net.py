import pandas as pd
from constants import NODES_PATH, LINES_PATH, V_NOM, S_NOM, BUILDINGS, PV_NODES, ESS_NODES

def create_network():

    # Load data from Excel files
    nodes_data = pd.read_excel(NODES_PATH)
    lines_data = pd.read_excel(LINES_PATH)

    # Process node data
    nodes = list(nodes_data.loc[i, 'NODES'] for i in nodes_data.index)
    print(nodes)
    node_types = {nodes_data.loc[i, 'NODES']: nodes_data.loc[i, 'Tb'] for i in nodes_data.index}
    p_demand = {nodes_data.loc[i, 'NODES']: nodes_data.loc[i, 'PDn'] / S_NOM for i in nodes_data.index}
    print(p_demand)
    q_demand = {nodes_data.loc[i, 'NODES']: nodes_data.loc[i, 'QDn'] / S_NOM for i in nodes_data.index}

    # Process line data
    lines = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']) for i in lines_data.index}
    r = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']): lines_data.loc[i, 'R'] / (V_NOM**2 * 1000 / S_NOM) for i in lines_data.index}
    x = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']): lines_data.loc[i, 'X'] / (V_NOM**2 * 1000 / S_NOM) for i in lines_data.index}
    i_max = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']): lines_data.loc[i, 'Imax'] / (S_NOM / V_NOM) for i in lines_data.index}

    # Return a structured dictionary with network data
    return {
        'bus_numbers': nodes,
        'line_connections': list(lines),
        'line_resistances': r,
        'line_reactances': x,
        'max_line_currents': i_max,
        'bus_types': node_types,
        'active_power_demand': p_demand,
        'reactive_power_demand': q_demand,
        'buildings': BUILDINGS,
        'PVs_at_buildings': PV_NODES,
        'ESSs_at_buildings': ESS_NODES
    }