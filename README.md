# CEA

This is the source code for ICDE 2020 paper Collective Embedding-based Entity Alignment via Adaptive Features (CEA)
https://arxiv.org/abs/1912.08404


The datasets are obtained from BootEA and RSNs.

stringsim.py generates string similarity matrix between entity names.

main.py generates the alignment accuracy. 

Before running main.py, you need to generate name_vec.txt, the entity name embeddings for entities in each dataset. 
It should be placed under the directory of each dataset.
The format of name_vec.txt is 
[entity id]\t[entity identifier]\t[embedding vectors seperated by space]
