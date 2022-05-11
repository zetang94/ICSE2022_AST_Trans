import argparse
import os
from pathlib import Path

from py_config_runner import ConfigObject
from ax.service.managed_loop import optimize
from script import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Example application")
    parser.add_argument("--config", type=Path, help="Input configuration file")
    parser.add_argument('--use_hype_params', action='store_true')
    parser.add_argument('--data_type', type=str, default='')
    parser.add_argument('--g', type=str, default='')
    args = parser.parse_args()

    assert args.config is not None
    assert args.config.exists()

    config = ConfigObject(args.config)
    print(1)
    if args.g != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.g
        config.device = "cuda"
        config.g = args.g
        if len(args.g.split(',')) > 1:
            config.multi_gpu = True
            config.batch_size = config.batch_size * len(args.g.split(','))
        else:
            config.multi_gpu = False
    else:
        config.device = 'cpu'
        config.multi_gpu = False

    if args.use_hype_params:
        config.hype_parameters = [
            {
                "name": "max_par_rel_pos",
                "type": "choice",
                "values": [1, 5, 10],
                "value_type": "int"
            },
            {
                "name": "max_bro_rel_pos",
                "type": "choice",
                "values": [1, 3, 5],
                "value_type": "int"
            },
            {
                "name": "par_heads",
                "type": "choice",
                "values": [0, 2, 4, 6, 8],
                "value_type": "int"
             }

        ]

        config.hype_parameters = {
            'max_par_rel_pos': 5,
            'max_bro_rel_pos': 5,
            'par_heads': 4
        }
        run(config, config.hype_parameters)
    else:
        if args.data_type != '':
            config.data_type = args.data_type
            config.task_name += args.data_type
        run(config)


