import sys
import yaml
import importlib
from pprint import pprint
from core.utils import param_handling
#from exp_gtzan_selframes.exp_gtzan_selframes import run, parse_commandline
#from exp_gtzan_selframes.exp_gtzan_selframes_norm_feats import run, parse_commandline

if __name__ == "__main__":
    
    args = sys.argv

    if len(args) < 3:
        print('Usage: %s experiment_file_module parameters.yaml [experiment_parameters]' % args[0])
        exit(1)

    params = yaml.load(open(args[2]))

    params['script_args'] = args

    exp_module = args[1].replace("/", ".").replace('.py', '')

    mod = importlib.import_module(exp_module)

    parse_commandline = eval("mod.parse_commandline")
    run = eval("mod.run")

    ovw = parse_commandline(args)
    ovw.extend(param_handling.parse_commandline(args))
    
    param_handling.overwrite_params(params, ovw)

    pprint(params)

    run(params)

