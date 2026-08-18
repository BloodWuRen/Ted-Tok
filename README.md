# Ted-Tok: Maintaining an Evolving Vocabulary for Lifelong Learning

This repository contains the official code for the ACL 2026 paper
[Ted-Tok: Maintaining an Evolving Vocabulary for Lifelong Learning](https://aclanthology.org/2026.acl-long.394/).

Ted-Tok studies a practical limitation of lifelong language learning: model
weights are usually updated as new data arrives, but the tokenizer vocabulary is
kept fixed. As language changes over time, a static tokenizer increasingly
fragments new lexical items, reduces compression efficiency, and can hurt
downstream performance. Ted-Tok maintains an evolving BPE vocabulary during
training by estimating token and token-pair frequencies online, deleting
outdated sink tokens, and adding new merge rules for emerging patterns.

## Repository Structure

```text
Ted-Tok/
|-- tokenizer_module/         # Shared BPE tokenizer builder
|-- SA/                       # Synthetic arithmetic experiment
|-- QA/                       # Lifelong news-language/QA experiment with nanoGPT
`-- MT/                       # WMT machine translation experiment based on verl
```

Each experiment directory contains its own README with more detailed commands.
The top-level README gives the common workflow and entry points.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{huang-etal-2026-ted,
    title = "Ted-Tok: Maintaining an Evolving Vocabulary for Lifelong Learning",
    author = "Huang, Jiameng  and
      Zhang, Zhi  and
      He, Zhenyu  and
      Sun, Jiacheng  and
      He, Di",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.394/",
    doi = "10.18653/v1/2026.acl-long.394",
    pages = "8706--8719",
    ISBN = "979-8-89176-390-6"
}
```
