# Towards Fair Graph Federated Learning via Incentive Mechanisms
### About
This is the Pytorch implementation of the paper "[Towards Fair Graph Federated Learning via Incentive Mechanisms](http://arxiv.org/abs/2312.13306)" accepted by AAAI-2024.
### Setup
The script has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):


`pytorch == 1.8.0`

`pytorch-cluster == 1.5.9`

`pytorch-geometric == 2.0.4`

`pytorch-scatter == 2.0.9`

`pytorch-sparse == 0.6.12`

`networkx == 2.8.7`

`scikit-learn == 1.1.2`



Or, you can install the dependency packages with the following command:



```
pip3 install -r requirements.txt
```
### Dataset
For the graph classiciation datasets, download and unzip it, and put it under data/.

### Files
`client.py`: the functions of the agents, such as updating the model locally and uploading the prototypes.

`server.py`: the functions of the server, such as aggregating the prototypes the contribution of the agents.

``



### Usage: How to run the code
```
python main_oneDS.py --repeat {index of the repeat}
      --data_group {dataset}
      --num_clients {num of clients}
      --seed {random seed}
      --lambda {coefficient of regularization term}
      --alpha {size of motif vocabulary}
Usage:
--repeat     int, the number to conduct the experiments
--data_group     str, the name of the dataset
--num_clients     int, number of clients
--seed     int, random seed of the experiments
--lambda     float, the coefficient of the regularization term
--alpha     float, the ratio of the motif vocabulary size in the entire motif vocabulary 
```
### Run repetitions for all datasets

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

### Contact
If you have any questions, feel free to contact me through email (chenglupan@zju.edu.cn).

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
