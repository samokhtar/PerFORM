# PerFORM: Implicit Neural Representations for Surrogate Modeling in the Built Environment

<p align="center">
<img src="docs/figures/abst.png" alt="abst" width="80%"/>
</p>

This repository provides the supporting code for **PerFORM: Implicit Neural Representations for Surrogate Modeling in the Built Environment**, a framework that leverages INRs for the coupled representation and predictive modeling of computationally prohibitive and/or non-differentiable geometric and physical quantities for the built environment.

<p align="center">
<img src="docs/figures/fig1.png" alt="fig1" width="80%"/>
</p>

<p align="center">
<img src="docs/figures/fig3.png" alt="fig3" width="80%"/>
</p>

<p align="center">
<img src="docs/figures/fig10.png" alt="fig10" width="80%"/>
</p>


## Repository Contents

The repository contains the code for models, architectural modules and training utilities needed to train, test, evaluate and visualize the frameworks presented in the paper. It follows the following structure

```
PerFORM/
├── src/
|   |                           # dataset processing and visualization code
│   ├── learning/               # models, modules, training utils
│   ├── prism_morphology/       # morphology metrics 
│   └── scripts/                # models, modules, training utils
│     ├── geo/                  # scripts for geometry models
|     └── perf/                 # scripts for performance models
├── requirements.txt
├── LICENSE.md
└── README.md
```

- All **our code** lives under `src/`

## Citation

When building on this research or using this code, please cite using the following:

```
@article{mokhtarImplicitNeuralRepresentations2025,
  title = {Implicit Neural Representations for Surrogate Modeling in the Built Environment},
  author = {Mokhtar, Sarah and Mueller, Caitlin},
  date = {2025},
  journaltitle = {...},
  shortjournal = {...}
}
```









