# CEA

This is the source code for ICDE 2020 paper Collective Embedding-based Entity Alignment via Adaptive Features ([CEA](https://arxiv.org/abs/1912.08404)).

The datasets are obtained from [BootEA](https://github.com/nju-websoft/BootEA) and [RSN](https://github.com/nju-websoft/RSN).

stringsim.py generates string similarity matrix between entity names.

main.py generates the alignment accuracy. 

Before running main.py, you need to generate name_vec.txt, the entity name embeddings for entities in each dataset. 
It should be placed under the directory of each dataset.
The format of name_vec.txt is 
```
[entity id]\t[entity identifier]\t[embedding vectors seperated by space]
```

You can also find our generated entity name embeddings in the zip file.

If you find our work useful, please kindly cite it as follows:
```
@inproceedings{CEA,
	Author = {Weixin Zeng and Xiang Zhao and Jiuyang Tang and Xuemin Lin},
	Booktitle = {ICDE 2020},
	Title = {Collective Embedding-based Entity Alignment via Adaptive Features},
	Year = {2020}
}
```
