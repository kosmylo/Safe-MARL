from create_net import create_network
from opf import opf_model
import logging

logger = logging.getLogger(__name__)

# Load the network data
network_data = create_network()

results = opf_model(network_data)