from common.arguments import get_train_args
from train import Trainer
from printcompare import *
import sys


args, ENV, Chip, Manager = get_train_args()
args.model_dir='./model/oc{}'.format(args.oc)
args.result_dir='./TrainResult/oc{}'.format(args.oc)
all_data =[]

for i in range(0,6):
    args.ith_run = i
    nargs=args.netdata
    manager = Manager(Chip(args.width, args.length), task=args.task, fov=[nargs[args.task].fov,nargs[args.task].fov], oc=args.oc,
                      stall=args.stall)
    env = ENV(manager)
    print(args)
    args.__dict__.update(env.get_env_info())
    # args.obs_shape = args.obs_shape[args.task]
    runner = Trainer(env, args)
    record = runner.run()# 不在线评估
    all_data.append(record)
    np.save(args.model_dir + '/vdn/fov{}/task{}/'.format(nargs[args.task].fov, args.task)+'tranfile56789', all_data)

if args.max_n_drop is not None:
    dropnums = range(3, args.max_n_drop + 1)
    total_data = processdata(all_data, dropnums, f'task{args.task+1}')
    for d in dropnums:
        fig = pltsame(total_data[d], d)
        fig.savefig(runner.save_path + f'/pltddrop_{d}.png',
                    bbox_inches='tight')
