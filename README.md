This is the code repository for the paper


&nbsp;&nbsp;&nbsp;&nbsp; **[CycleGT: Unsupervised Graph-to-Text and Text-to-Graph Generation via Cycle Training](https://arxiv.org/pdf/2006.04702.pdf)** 

### Dependencies
- pytorch 1.4.0 cu10
- pycocoevalcap python3 version
- `pip install -r requirements.txt`

### How to Run
```
python -u main.py
```

Feel free to change [config.yaml](config.yaml) to set other configurations.



## Results on WebNLG
|Method| BLEU | Micro F1 | Macro F1|
|-|-|-|-|
|Back Translation w/ warmup| 43.08+-0.39| 62.3+-0.2 | 52.1+-0.3|
|Back Translation w/o warmup| 44.97+-0.66| 61.6+-0.2 | 51.4+-0.2|
|Supervised | 44.88+-0.66| 61.0+-0.3| 51.0+-0.3|

## Citing CycleGT

If you use CycleGT, please cite **[CycleGT: Unsupervised Graph-to-Text and Text-to-Graph Generation via Cycle Training](https://arxiv.org/pdf/2006.04702.pdf)**.

```bibtex
@article{guo2020cyclegt,
  author    = {Qipeng Guo and Zhijing Jin and Xipeng Qiu and Weinan Zhang and David Wipf and Zheng Zhang},
  title     = {CycleGT: Unsupervised Graph-to-Text and Text-to-Graph Generation via Cycle Training},
  journal   = {CoRR},
  volume    = {abs/2006.04702},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.04702},
  archivePrefix = {arXiv},
  eprint    = {2006.04702}
}
```