import torch
import argparse
import yaml
import pickle

from madrl.models.model_registry import Model, Strategy
from madrl.environments.flex_provision.flexibility_provision_env import FlexibilityProvisionEnv
from utils.util import convert
from utils.tester import PGTester

parser = argparse.ArgumentParser(description="Test rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?", default="maddpg", help="Please enter the alg name.")
parser.add_argument("--env", type=str, nargs="?", default="flex_provision", help="Please enter the env name.")
parser.add_argument("--test-mode", type=str, nargs="?", default="single", help="Please input the valid test mode: single or batch.")
parser.add_argument("--test-day", type=int, nargs="?", default=730, help="Please input the day you would test if the test mode is single.")
parser.add_argument("--render", action="store_true", help="Activate the rendering of the environment.")
argv = parser.parse_args()

# load env args
with open("./madrl/args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
env_config_dict["data_path"] = "/".join(data_path)

# for one-day test
env_config_dict["episode_limit"] = 480

# load default args
with open("./madrl/args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)
default_config_dict["max_steps"] = 480

# load alg args
with open("./madrl/args/alg_args/"+argv.alg+".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_low"] = env_config_dict.get("action_low", 0.0)
    alg_config_dict["action_high"] = env_config_dict.get("action_high", 1.0)
    alg_config_dict["action_bias"] = env_config_dict.get("action_bias", 0.0)  
    alg_config_dict["action_scale"] = env_config_dict.get("action_scale", 1.0)

log_name = "-".join([argv.env, argv.alg])
alg_config_dict = {**default_config_dict, **alg_config_dict}

# define envs
env = FlexibilityProvisionEnv(env_config_dict)

alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["action_dim"] = env.get_total_actions()
alg_config_dict["cuda"] = False
args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] == "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

LOAD_PATH = save_path+"model_save/"+log_name+"/model.pt"

model = Model[argv.alg]

strategy = Strategy[argv.alg]

if args.target:
    target_net = model(args)
    behaviour_net = model(args, target_net)
else:
    behaviour_net = model(args)
checkpoint = torch.load(LOAD_PATH, map_location='cpu') if not args.cuda else torch.load(LOAD_PATH)
behaviour_net.load_state_dict(checkpoint['model_state_dict'])

print (f"{args}\n")

if strategy == "pg":
    test = PGTester(args, behaviour_net, env, argv.render)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

if argv.test_mode == 'single':
    # record = test.run(199, 23, 2) # (day, hour, 3min)
    # record = test.run(730, 23, 2) # (day, hour, 3min)
    record = test.run(argv.test_day, 23, 2)
    with open('test_record_'+log_name+f'_day{argv.test_day}'+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)
elif argv.test_mode == 'batch':
    record = test.batch_run(10)
    with open('test_record_'+log_name+'_'+argv.test_mode+'.pickle', 'wb') as f:
        pickle.dump(record, f, pickle.HIGHEST_PROTOCOL)