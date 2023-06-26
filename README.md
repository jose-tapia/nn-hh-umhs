# Manuscript 2494 - Result Datasets & Notebook

This repository contains the resulting datasets of the work entitled **A Primary Study on Hyper-Heuristics Powered by Artificial Neural Networks for Customising Population-based Metaheuristics in Continuous Optimisation Problems** presented in *CEC-22*. We also included the main jupyter file to plot the figures thereby presented.

Authors: _José Manuel Tapia-Avitia, Jorge M. Cruz-Duarte, Ivan Amaya, José Carlos Ortiz-Bayliss, Hugo Terashima Marín, Nelishia Pillay_

## Important

Due to the file size limitation of GitHub, we provided the resulting data files in split zip files into [data_files](./data_files): `exp_output.zip`, `exp_output.z01`, `exp_output.z02`. So, after cloning this repository, you must follow the steps below in your terminal.
1. Go to the `tl-hh-umhs/data_files` folder:
```shell
cd data_files
```
2. Combine the split zip files:
```shell
zip -F exp_output.zip --out exp_output-single.zip`
```
3. Unzip the combined zip file:
```shell
unzip exp_output-single.zip
```
Then, you have the data result files.

## Requirements
- Python v3.7+
- [CUSTOMHyS framework](https://github.com/jcrvz/customhys.git)
- Standard modules: os, matplotlib, seaborn, numpy, pandas, scipy.stats, tensorflow

## Files
- **Main notebook**: [processing_results_nnhh.ipynb](./processing_results_nnhh.ipynb)
- Results from the proposed approaches (_this folder will be available once unzip exp_output files_): [data_files/exp_output](./data_files/exp_output)
- Raw figures generated from the main notebook: [data_files/exp_figures](./data_files/exp_figures)
- Experimental configurations used in this work: [exconf](./exconf)
- Results for basic metaheuristics: [data_files/basic-metaheuristics-data_v2.json](./data_files/basic-metaheuristics-data_v2.json)
- Collection of basic metaheuristics: [collections/basicmetaheuristics.txt](./collections/basicmetaheuristics.txt)
- Collection of default heuristics: [collections/default.txt](./collections/default.txt)

## Contact information

José Manuel Tapia-Avitia - [A00834191@tec.mx](mailto:A00834191@tec.mx)
Jorge M. Cruz-Duarte - [jcrvz.co](https://jcrvz.co), [jorge.cruz@tec.mx](mailto:jorge.cruz@tec.mx)

Distributed under the MIT license. See [LICENSE](./LICENSE) for more information.

