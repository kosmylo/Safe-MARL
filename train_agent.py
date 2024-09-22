import torch as th
import os
import argparse
import yaml
from tensorboardX import SummaryWriter

from madrl.models.model_registry import Model, Strategy
from madrl.environments.flex_provision.flexibility_provision_env import FlexibilityProvisionEnv
from utils.util import convert, dict2str
from utils.trainer import PGTrainer
from utils.plot_res import plot_training_metrics
import time
import logging

# Set up logging directory and file
log_dir = "logs/train_logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "train_log.txt")

# Configure the logger
train_logger = logging.getLogger('TrainLogger')
train_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler(log_file_path, mode='w')
file_handler.setFormatter(formatter)
train_logger.addHandler(file_handler)

# Stream handler (optional, if you want to see logs in the console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
train_logger.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?", default="maddpg", help="Please enter the alg name.")
parser.add_argument("--env", type=str, nargs="?", default="flex_provision", help="Please enter the env name.")
argv = parser.parse_args()

# load env args
with open("./madrl/args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
env_config_dict["data_path"] = "/".join(data_path)
env_config_dict["alg"] = argv.alg

# load default args
with open("./madrl/args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)

# load alg args
with open("./madrl/args/alg_args/" + argv.alg + ".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_low"] = env_config_dict.get("action_low", 0.0)
    alg_config_dict["action_high"] = env_config_dict.get("action_high", 1.0)
    alg_config_dict["action_bias"] = env_config_dict.get("action_bias", 0.0)  
    alg_config_dict["action_scale"] = env_config_dict.get("action_scale", 1.0)

alg_config_dict["alg"] = argv.alg

log_name = "-".join([argv.env, argv.alg])
alg_config_dict = {**default_config_dict, **alg_config_dict}

# define envs
env = FlexibilityProvisionEnv(env_config_dict)

alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["state_size"] = env.get_state_size()
alg_config_dict["action_dim"] = env.get_total_actions()
args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] == "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

# create the save folders
if "model_save" not in os.listdir(save_path):
    os.mkdir(save_path + "model_save")
if "tensorboard" not in os.listdir(save_path):
    os.mkdir(save_path + "tensorboard")
if log_name not in os.listdir(save_path + "model_save/"):
    os.mkdir(save_path + "model_save/" + log_name)
if log_name not in os.listdir(save_path + "tensorboard/"):
    os.mkdir(save_path + "tensorboard/" + log_name)
else:
    path = save_path + "tensorboard/" + log_name
    for f in os.listdir(path):
        file_path = os.path.join(path,f)
        if os.path.isfile(file_path):
            os.remove(file_path)

# create the logger
logger = SummaryWriter(save_path + "tensorboard/" + log_name)

model = Model[argv.alg]

strategy = Strategy[argv.alg]

train_logger.info(f"{args}\n")

if strategy == "pg":
    train = PGTrainer(args, model, env, logger)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

with open(save_path + "tensorboard/" + log_name + "/log.txt", "w+") as file:
    alg_args2str = dict2str(alg_config_dict, 'alg_params')
    env_args2str = dict2str(env_config_dict, 'env_params')
    file.write(alg_args2str + "\n")
    file.write(env_args2str + "\n")

# Initialize variables to track training metrics
rewards = []
policy_losses = []
value_losses = []
start_time = time.time()

for i in range(args.train_episodes_num):
    stat = {}
    train_logger.info(f"Running episode {i}")
    train.run(stat, i)
    train_logger.info(f"Episode {i} completed")

    episode_reward = stat.get('mean_train_reward', 0.0)
    rewards.append(episode_reward)
    
    # Use None if the loss isn't computed
    policy_loss = stat.get('mean_train_policy_loss', None)
    if policy_loss is not None:
        policy_losses.append(policy_loss)

    value_loss = stat.get('mean_train_value_loss', None)
    if value_loss is not None:
        value_losses.append(value_loss)
    
    train.logging(stat)
    if i%args.save_model_freq == args.save_model_freq-1:
        train.print_info(stat)
        th.save({"model_state_dict": train.behaviour_net.state_dict()}, save_path + "model_save/" + log_name + "/model.pt")
        train_logger.info("The model is saved!\n")

# Calculate total training time
end_time = time.time()
total_training_time = end_time - start_time
train_logger.info(f"Total Training Time: {total_training_time:.2f} seconds")

plot_training_metrics(rewards, policy_losses, value_losses)

logger.close()