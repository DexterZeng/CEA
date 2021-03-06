# CEA

This is the source code for ICDE 2020 paper Collective Entity Alignment via Adaptive Features ([CEA](https://arxiv.org/abs/1912.08404)).

The code is based on the old version of [GCN-Align](https://github.com/1049451037/GCN-Align). 
The datasets are obtained from [BootEA](https://github.com/nju-websoft/BootEA) and [RSN](https://github.com/nju-websoft/RSN).

stringsim.py generates string similarity matrix between entity names.

main.py generates the alignment results. 

Before running main.py, you need to generate name_vec.txt, the entity name embeddings for entities in each dataset. 
It should be placed under the directory of each dataset.
The format of name_vec.txt is 
```
[entity id]\t[entity identifier]\t[embedding vectors seperated by space]
```

If you want to use the entity name embeddings in our paper, please download from [here](https://share.weiyun.com/5qxLmEI). Note that for DBP15K datasets, the name embeddings should be read in the following way (similar to [RDGCN](https://github.com/StephanieWyt/RDGCN)):
```
with open(file='./data/' + Config.language + '/' + Config.language.split('_')[0] + '_vectorList.json',
	  mode='r', encoding='utf-8') as f:
    embedding_list = json.load(f)
    print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    ne_vec = np.array(embedding_list)
```	    


If you find our work useful, please kindly cite it as follows:
```
@inproceedings{DBLP:conf/icde/Zeng0T020,
  author    = {Weixin Zeng and
               Xiang Zhao and
               Jiuyang Tang and
               Xuemin Lin},
  title     = {Collective Entity Alignment via Adaptive Features},
  booktitle = {36th {IEEE} International Conference on Data Engineering, {ICDE} 2020,
               Dallas, TX, USA, April 20-24, 2020},
  pages     = {1870--1873},
  publisher = {{IEEE}},
  year      = {2020},
}
```
