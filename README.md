# CycleGT
code of CycleGT

Dependencies  
pytorch 1.4.0 cu10  
transformers 2.3.0  
pycocoevalcap python3 version  
sklearn 0.22  
dgl 0.4.2  

Run
python -u main.py

change mode in config.yaml to get different settings

Results on WebNLG
|Method| BLEU | Micro F1 | Macro F1|
|-|-|-|-|
|Back Translation w/ warmup| 43.08+-0.39| 62.3+-0.2 | 52.1+-0.3|
|Back Translation w/o warmup| 44.97+-0.66| 61.6+-0.2 | 51.4+-0.2|
|Supervised | 44.88+-0.66| 61.0+-0.3| 51.0+-0.3|
