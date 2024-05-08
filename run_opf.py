from utils.create_net import create_network
from OPF.opf import opf_model
from OPF.plot_res import plot_optimization_results
import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Load the network data
network_data = create_network()

results = opf_model(network_data)

logger.info('Optimization results: %s', results)

plot_optimization_results(results)