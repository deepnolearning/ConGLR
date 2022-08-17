Please install relevant python packages before running the code. Our settings are as follows:

```
dgl==0.4.2
lmdb==0.98
networkx==2.4
scikit-learn==0.22.1
torch==1.9.0
tqdm==4.43.0
```

In the folder "src", we provide a python script "run_bash.py" for quick running with default hyper-parameters to train or test as follows:

```
def train():
    gpu = 3
    lr = 0.001  # 001
    l2 = 0.0005  # 0005
    epoch = 10
    # dataset_list = ['fb237','WN18RR','nell']
    # version = [1,2,3,4]

    dataset_list = ['WN18RR']
    version = [1]
    for dataset in dataset_list:
        for v in version:
            temp_cmd = f'python train.py -d {dataset}_v{v} -e {dataset}_v{v} --gpu {gpu} 												--lr {lr} --l2 {l2} --num_epochs {epoch}'
            os.system(temp_cmd)
```

```
def test():
    gpu = 0
    num_neg_samples_per_link = 50  # 1
    # dataset_list = ['fb237','WN18RR','nell']
    # version = [1,2,3,4]

    dataset_list = ['WN18RR']
    version = [1]
    for dataset in dataset_list:
        for v in version:
            temp_cmd = f'python test_auc.py -d {dataset}_v{v}_ind -e {dataset}_v{v} 													--gpu {gpu} --num_neg_samples_per_link {num_neg_samples_per_link}'
            os.system(temp_cmd)
```

When conducting training or testing during the experiment, you only need to convert to the corresponding functions of "train()" or "test()". The "num_neg_samples_per_link" in function "test()" means the number of negative samples during testing, which has two values (1 for the classification task and 50 for the ranking task) in our experiments. If you want to attempt other hyper-parameters, you can add the argument descriptions in function "train()" or change the corresponding values in "train.py".

Some codes are referenced by [GraIL](https://github.com/kkteru/grail) and [CoMPILE](https://github.com/TmacMai/CoMPILE_Inductive_Knowledge_Graph).
**If the code is useful for you, please cite the following paper:**
```
@inproceedings{DBLP:conf/sigir/LinLXPZZZ22,
  author    = {Qika Lin and
               Jun Liu and
               Fangzhi Xu and
               Yudai Pan and
               Yifan Zhu and
               Lingling Zhang and
               Tianzhe Zhao},
  title     = {Incorporating Context Graph with Logical Reasoning for Inductive Relation Prediction},
  booktitle = {The 45th International {ACM} {SIGIR} Conference on Research and Development in Information Retrieval},
  pages     = {893--903},
  publisher = {{ACM}},
  year      = {2022}
}
```
