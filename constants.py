NODES_PATH = 'Nodes_33.xlsx'
LINES_PATH = 'Lines_33.xlsx'

V_NOM = 12.66  # kV
S_NOM = 1000  # kVA
V_MIN = 0.9  # pu
V_MAX = 1.1 # pu

BUILDINGS = [5, 10, 15, 20, 25] 
PV_NODES = [5, 10, 15, 20, 25] 
ESS_NODES = [5, 10, 15, 20, 25]

PV_CAPACITY = 50 / S_NOM

T = 24

MAX_POWER_REDUCTION_PERCENT = 0.50  # 50% reduction

COS_PHIMAX = 0.95 # Maximum power factor

FLEX_PRICE = {i: 0.20 if 17 <= i <= 21 else 0.10 for i in range(1, 25)} # Flexibility price ($/kWh)
PV_COST = {der: 0.05 for der in PV_NODES} # Cost of reactive power control of each PV ($/kWh)
ESS_COST = {ess: 0.03 for ess in ESS_NODES}  # Cost of operating each ESS ($/kWh)
DISCOMFORT_COEFF = {building: 0.15 for building in BUILDINGS}  # Discomfort cost coefficients ($/kWh)

ETA_CH = 0.9
ETA_DIS = 0.9
E_MIN = 0 
E_MAX = 25 / S_NOM 
P_CH_MAX = 5 / S_NOM
P_DIS_MAX = 5 / S_NOM 

LOAD_PROFILE = [0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
PV_PROFILE = [0, 0, 0, 0.1, 0.3, 0.5, 0.8, 0.9, 1.0, 0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]