# Towards Fair Graph Federated Learning via Incentive Mechanisms
### About
This is the Pytorch implementation of the paper "[Towards Fair Graph Federated Learning via Incentive Mechanisms](http://arxiv.org/abs/2312.13306)" accepted by AAAI-2024.
### Setup
```
pip3 install -r requirements.txt
```
### Dataset
For the graph classiciation datasets, download and unzip it, and put it under data/.
### Usage: How to run the code
```
python main_oneDS.py --repeat {index of the repeat}
      --data_group {dataset}
      --num_clients {num of clients}
      --seed {random seed}
      --lambda {coefficient of regularization term}
      --alpha {size of motif vocabulary}
```
## Run repetitions for all datasets

To averagely aggregate all repetitions, and get the overall performance:

```
python aggregateResults.py --inpath {the path to repetitions} --outpath {the path to outputs} --data_partition {the data partition mechanism}
```

Or, to run one file for all:

```
bash runnerfile_aggregateResults
```



### Acknowledgement
Some of the implementation is adopted from [Federated Graph Classification over Non-IID Graphs](https://github.com/Oxfordblue7/GCFL).

### Cite
If you find this work helpful, please cite
```
@inproceedings{Pan2023TowardsFG,
  title={Towards Fair Graph Federated Learning via Incentive Mechanisms},
  author={Chenglu Pan and Jiarong Xu and Yue Yu and Ziqi Yang and Qingbiao Wu and Chunping Wang and Lei Chen and Yang Yang},
  year={2024},
  booktitle={AAAI}
}
```
