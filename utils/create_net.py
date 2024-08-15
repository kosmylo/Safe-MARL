import pandas as pd
import yaml

# load env args
with open("./MADRL/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]

def create_network():

    # Load data from Excel files
    nodes_data = pd.read_excel(f"{env_config_dict['data_path']}/Nodes_33.xlsx")
    lines_data = pd.read_excel(f"{env_config_dict['data_path']}/Lines_33.xlsx")
    
    # Process node data
    nodes = list(nodes_data.loc[i, 'NODES'] for i in nodes_data.index)
    node_types = {nodes_data.loc[i, 'NODES']: nodes_data.loc[i, 'Tb'] for i in nodes_data.index}
    p_demand = {nodes_data.loc[i, 'NODES']: nodes_data.loc[i, 'PDn'] / env_config_dict['s_nom'] for i in nodes_data.index}
    q_demand = {nodes_data.loc[i, 'NODES']: nodes_data.loc[i, 'QDn'] / env_config_dict['s_nom'] for i in nodes_data.index}

    # Process line data
    lines = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']) for i in lines_data.index}
    r = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']): lines_data.loc[i, 'R'] / (env_config_dict['v_nom']**2 * 1000 / env_config_dict['s_nom']) for i in lines_data.index}
    x = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']): lines_data.loc[i, 'X'] / (env_config_dict['v_nom']**2 * 1000 / env_config_dict['s_nom']) for i in lines_data.index}
    i_max = {(lines_data.loc[i, 'FROM'], lines_data.loc[i, 'TO']): lines_data.loc[i, 'Imax'] / (env_config_dict['s_nom'] / env_config_dict['v_nom']) for i in lines_data.index}

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
        'buildings': env_config_dict['buildings'],
        'PVs_at_buildings': env_config_dict['pv_nodes'],
        'ESSs_at_buildings': env_config_dict['ess_nodes']
    }