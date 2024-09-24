# Safe-MARL

Safe-MARL is a project focused on implementing Safe Multi-Agent Reinforcement Learning (MARL) for flexibility provision in distribution networks. The goal is to develop and test MARL algorithms that can manage energy resources of a building while ensuring safety constraints are met.

Features
--------

-   **Safe Multi-Agent Reinforcement Learning**: Implementation of various MARL algorithms with safety considerations.
-   **Flexibility Provision Environment**: Custom environment simulating a distribution network for flexibility management tasks.
-   **Data-Driven Simulation**: Utilizes data for load, PV generation, and flexibility prices.
-   **Visualization Tools**: Scripts for plotting results and analyzing agent performance.
-   **Logging and Monitoring**: Integration with TensorBoard for tracking training progress.

Installation
------------

### Prerequisites

-   Python 3.7 or higher
-   pip
-   Git

### Clone the Repository

1. `git clone https://github.com/kosmylo/Safe-MARL.git`
2. `cd Safe-MARL`

### Create a Virtual Environment

1. `python -m venv venv`
2. `source venv/bin/activate`       

### Install Dependencies

`pip install -r requirements.txt`

Getting Started
---------------

### Setting Up the Environment

The project uses a custom environment for flexibility provision in distribution grids, defined in `madrl/environments/flex_provision/`.

### Data Preparation

Place your data files in the `data/` directory. The project expects data files like:

-   `load_active.csv`
-   `load_reactive.csv`
-   `pv_active.csv`
-   `prices.csv`

Usage
-----

### Training Agents

To train agents using one of the MARL algorithms:

`python train_agent.py`

### Testing Agents

To test a trained agent:

`python test_agent.py`

### Running Environment Simulations

To run the environment without training:

`python run_env.py`

### Running Power Flow Analysis

To perform power flow analysis:

`python run_pf.py`

### Running Optimal Power Flow (OPF)

To run OPF calculations:

`python run_opf.py`

Data
----

The `data/` directory contains all necessary data files:

-   **Lines_33.xlsx** and **Nodes_33.xlsx**: Define the network topology.
-   **load_active.csv**, **load_reactive.csv**: Load profiles.
-   **pv_active.csv**: PV generation profiles.
-   **prices.csv**: Flexibility prices.
