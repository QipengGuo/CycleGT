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

Feel free to change [f_config.yaml](f_config.yaml) to set other configurations.

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