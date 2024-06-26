import os
import argparse
import random
import torch
from pathlib import Path
import copy

import setupGC
from training import *


def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcfl(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcfl_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_GC{suffix}.csv')

    frame = run_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplus(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}.csv')

    frame = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplusdWs(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus with dWs ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplusDWs_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplusDWs_GC{suffix}.csv')

    frame = run_gcflplus_dWs(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_prototype(clients, server):
    print("\nDone setting up prototype devices.")
    print("Running fed prototype ...")
    frame = run_prototype(clients, server, args.num_rounds, args.local_epoch, samp=None)
    #outfile = os.path.join()

def process_protoreput(clients, server):
    print("\nDone setting up prototype devices.")
    print("Running fed prototype ...")
    frame = run_protoreput(clients, server, args.num_rounds, args.device, samp=None)


def process_protoreput2(clients, server):
    print("\nDone setting up prototype devices.")
    print("Running fed prototype ...")
    frame = run_protoreput2(clients, server, args.num_rounds, args.device, args.disable_dp, samp=None)

# def process_protoreput4(clients, server):
#     print("\nDone setting up prototype devices.")
#     print("Running fed prototype ...")
#     frame = run_protoreput4(clients, server, args.num_rounds, args.device, args.disable_dp, samp=None)





def process_protoreput3(clients, server):
    print("\nDone setting up prototype devices.")
    print("Running fed prototype ...")
    frame = run_protoreput3(clients, server, args.num_rounds, args.device, samp=None)

def process_reput(clients, server):
    print('\nDone setting up devices.')
    print('Running fed reput ...')
    frame = run_reput(clients, server, args.num_rounds, args.local_epoch, samp=None)

def process_reput2(clients, server):
    print('\nDone setting up devices.')
    print('Running fed reput ...')
    frame = run_reput2(clients, server, args.num_rounds, args.local_epoch, samp=None)

def process_reput3(clients, server):
    print('\nDone setting up devices.')
    print('Running fed reput ...')
    frame = run_reput3(clients, server, args.num_rounds, 1, samp=None)






if __name__ == '__main__':








    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='CPU / GPU device.')
    parser.add_argument('--num_repeat', type=int, default=5,
                        help='number of repeating rounds to simulate;')
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of GINconv layers')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=123)

    parser.add_argument('--datapath', type=str, default='./data',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='IMDB-BINARY')

    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--num_clients', help='number of clients',
                        type=int, default=10)
    parser.add_argument('--overlap', help='whether clients have overlapped data',
                        type=bool, default=False)
    parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                        type=bool, default=False)
    parser.add_argument('--seq_length', help='the length of the gradient norm sequence',
                        type=int, default=5)
    parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',
                        type=float, default=0.03)
    parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',
                        type=float, default=0.06)
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.85)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--disable_dp', type=bool, default=False)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2

    outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    if args.overlap and args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/oneDS-overlap")
    elif args.overlap:
        outpath = os.path.join(outbase, f"oneDS-overlap")
    elif args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/oneDS-nonOverlap")
    else:
        outpath = os.path.join(outbase, f"oneDS-nonOverlap")
    outpath = os.path.join(outpath, f'{args.data_group}-{args.num_clients}clients', f'eps_{EPS_1}_{EPS_2}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    """ distributed one dataset to multiple clients """

    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")

    if args.repeat is not None:
        Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)
    
    splitedData, df_stats = setupGC.prepareData_oneDS(args.datapath, args.data_group, num_client=args.num_clients, batchSize=args.batch_size,
                                                      convert_x=args.convert_x, seed=seed_dataSplit, overlap=args.overlap, aug=args.aug)
    print("Done")

    # save statistics of data on clients
    if args.repeat is None:
        outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    else:
        outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    init_clients, init_server, init_idx_clients = setupGC.setup_devices(splitedData, args)
    print("\nDone setting up devices.")
    

    

    


    


    process_protoreput2(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    
    
    
