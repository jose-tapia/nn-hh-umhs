# Hyper-Heuristics Powered by Artificial Neural Networks for Cusomising Population-based Metahueristics in Continuous Optimisation Problems
## Result Datasets & Notebook

This repository contains the resulting datasets of using the hyper-heuristic based on neural networks to produce enhanced metahueristics. We also included the main jupyter file to plot the figures thereby presented.

Authors: _José Manuel Tapia-Avitia, Jorge M. Cruz-Duarte, Ivan Amaya, José Carlos Ortiz-Bayliss, Hugo Terashima Marín, Nelishia Pillay_

## Important

Due to the file size limitation of GitHub, we provided the resulting data files in split zip files into [data_files](./data_files): `all-exp-results.zip`, `all-exp-results.z01`, `all-exp-results.z02`, ..., `all-exp-results.z33`. So, after cloning this repository, you must follow the steps below in your terminal.

1. Go to the `tl-hh-umhs/data_files` folder:
```shell
cd data_files
```
2. Combine the split zip files:
```shell
zip -F all-exp-results.zip --out all-exp-results-single.zip`
```
3. Unzip the combined zip file:
```shell
unzip all-exp-results-single.zip
```
Then, you have the data result files.

## Requirements
- Python v3.8+
- [CUSTOMHyS framework](https://github.com/jcrvz/customhys.git)
- Standard modules: os, matplotlib, seaborn, numpy, pandas, scipy.stats, tensorflow

## Files
- **Main notebook**: [processing_results_nnhh.ipynb](./processing_results_nnhh.ipynb)
- Results from the proposed approaches (_this folder will be available once unzip all-exp-results files_): [data_files/all-exp-results](./data_files/all-exp-results)
- Raw figures generated from the main notebook: [data_files/exp_figures](./data_files/exp_figures)
- Experimental configurations used in this work: [exconf](./exconf)
- Results for basic metaheuristics: [data_files/basic-metaheuristics-data_v2.json](./data_files/basic-metaheuristics-data_v2.json)
- Collection of basic metaheuristics: [collections/basicmetaheuristics.txt](./collections/basicmetaheuristics.txt)
- Collection of default heuristics: [collections/default.txt](./collections/default.txt)

## Contact information

José Manuel Tapia-Avitia - [josetapia@exatec.tec.mx](mailto:josetapia@exatec.tec.mx)
Jorge M. Cruz-Duarte - [jcrvz.co](https://jcrvz.co), [jorge.cruz@tec.mx](mailto:jorge.cruz@tec.mx)

Distributed under the MIT license. See [LICENSE](./LICENSE) for more information.
