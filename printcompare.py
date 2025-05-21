from common.arguments import common_args, process_evaluate_args
import numpy as np
from agent.agent import Agents
from common.rollout import Evaluator
from copy import deepcopy
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def processdata(record, dropnums, dataname):
    total_data = {}
    for dropnum in dropnums:
        total_data[dropnum] = []
        for index in range(len(record)):
            r, step, cons, succ = record[index]
            df = pd.DataFrame({'Rewards': r[dropnum], 'Success rate': succ[dropnum], 'Completion steps': step[dropnum],
                               'Constraint': cons[dropnum], 'Train epoch': np.arange(0, len(r[dropnum])), 'repeat': index,
                               'schema': dataname})
            total_data[dropnum].append(df)
        total_data[dropnum] = pd.concat(total_data[dropnum], axis=0, ignore_index=True)
    return total_data

def extended_parser():
    parser = common_args()
    parser.add_argument('--ints', metavar='N', type=int, nargs='*', help='挑选哪一次或几次训练需要展示，默认就是文件夹里所有的模型')
    args = parser.parse_args()
    print(args.ints)
    return process_evaluate_args(args)


def pltsame(data, n, app=None):
    sns.set_theme(context='paper', style='whitegrid', font='Times New Roman', font_scale=1.8)
    fig1, ax = plt.subplots(1, 3, figsize=(15, 3), dpi=150)
    ax0 = sns.lineplot(x='Train epoch', y='Success rate', hue='schema', data=data, ax=ax[0])
    ax1 = sns.lineplot(x='Train epoch', y='Completion steps', hue='schema', data=data, ax=ax[1])
    ax2 = sns.lineplot(x='Train epoch', y='Constraint', hue='schema', data=data, ax=ax[2])

    if app is not None:
        x_app, y_app = app
        ax0.plot(x_app, y_app['success'], linestyle='--',
                 color=(0.2823529411764706, 0.47058823529411764, 0.8156862745098039))
        ax1.plot(x_app, y_app['step'], linestyle='--',
                 color=(0.2823529411764706, 0.47058823529411764, 0.8156862745098039))
    ax0.set_ylabel(r'$P_{success}$', fontsize=18)
    ax1.set_ylabel(r'$T_{avg}$', fontsize=18)
    ax2.set_ylabel('constraints', fontsize=18)
    ax0.set_yticks(np.arange(0, 1.1, 0.2))
    ax = plt.gca()
    h, l = ax.get_legend_handles_labels()
    ax2.legend(handles=h, labels=l)
    ax0.legend(loc=4, handles=h, labels=l)
    ax1.legend(loc='upper right', handles=h, labels=l)
    plt.suptitle('{} droplets'.format(n), x=0.11, fontdict={'family': 'Times New Roman', 'size': 18})
    plt.subplots_adjust(wspace=0.25)
    return fig1

def print_evaluate_for_drop_num():
    args, ENV, Chip, Manager = extended_parser()
    nargs = args.netdata
    save_path = args.result_dir + '/' + args.alg + '/fov{}/task{}'.format(nargs[args.task].fov, args.task) + '/CL'

    # ----一次运行FF
    assayManager = Manager(Chip(args.width, args.length), task=args.task, fov=[nargs[args.task].fov,nargs[args.task].fov],  oc=args.oc, stall=args.stall)
    env = ENV(assayManager)
    print(args)
    args.__dict__.update(env.get_env_info())

    if args.task ==0:
        dropnums= [3, 4, 5, 6, 8, 9, 10]
        chip_sizes = [10, 10, 15, 16, 18, 19, 20]
    if args.task==1:
        dropnums = [3, 4, 5]
        chip_sizes= [10, 10, 15]


    all_data =[]
    model_dir = args.model_dir + '/vdn/fov{}/task{}/'.format(nargs[args.task].fov, args.task)
    data_name = save_path + '/{}r,step,constraint,success'.format(dropnums)
    if not args.ints:
        # 找到文件所有的i值
        args.ints = set(
            int(filename.split('_')[0]) for filename in os.listdir(model_dir) if "rnn_net_params.pkl" in filename)
        print(args.ints)
    data_name = data_name + str(args.ints)
    if os.path.exists(data_name+'.npy'):
        all_data = np.load(data_name+'.npy', allow_pickle=True)
    else:
        print(data_name+'.npy','not exist')
        for i in args.ints:
            episode_rewards, episode_steps, episode_constraints, success_rate = {}, {}, {}, {}
            record = [episode_rewards, episode_steps, episode_constraints, success_rate]
            for drop_num in dropnums:
                for r in record:
                    r[drop_num] = []
            j = 0
            while j < 100: # 最多有99个文件
                filename = f"{i}_{j}_rnn_net_params.pkl" if j<99 else f"{i}_rnn_net_params.pkl"
                file_path = os.path.join(model_dir, filename)
                if not os.path.exists(file_path):
                    if j == 99:
                        break
                    j = 99
                    continue
                nargs[args.task].load_model_name = f"{i}_{j}_" if j<99 else f"{i}_"
                print(nargs[args.task].load_model_name)
                # print('chip_size updated:', 10)
                evaluator = Evaluator(env, Agents(args, task=args.task))
                for drop_num, size in zip(dropnums, chip_sizes):
                        # print('//drop_num:', drop_num,'chip_size updated:', w)
                    env.reset_chip(size, size)
                    onerecord= evaluator.evaluate(
                        args.evaluate_task, drop_num)
                    # print(drop_num, onerecord)
                    for _ in range(4):
                        record[_][drop_num].append(onerecord[_])
                    print('drop_num:', drop_num, 'step:', onerecord[1], 'constraints:', onerecord[2], 'success:', onerecord[3])
                j += 1
            all_data.append(record)
        np.save(data_name, all_data)
    total_data = processdata(all_data, dropnums, f'task{args.task+1}')

    for d in dropnums:
        fig = pltsame(total_data[d], d)
        fig.savefig(save_path + f'/pltddrop4t_{d}.png',
                    bbox_inches='tight')
        print(f'pltddrop4t_{d}.png saved at {save_path}')


if __name__ == '__main__':
    print_evaluate_for_drop_num()