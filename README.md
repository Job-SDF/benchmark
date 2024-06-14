<div align="center">
  <img src="figs/title.png" alt="Title Image">
    <p> 
    	<b>
        Job-SDF: A Multi-Granularity Dataset for Job Skill Demand Forecasting and Benchmarking <a href="" title="PDF">PDF</a>
        </b>
    </p>

---

<p align="center">
  <a href="## Overview">Overview</a> •
  <a href="## Installation">Installation</a> •
  <a href="## Dataset">Dataset</a> •
  <a href="## How to Run">How to Run </a> •
  <a href="## Directory Structure">Directory Structure</a> •
  <a href="## Citation">Citation</a> 
</p>
</div>

Official repository of paper [&#34;Job-SDF: A Multi-Granularity Dataset for Job Skill Demand Forecasting and Benchmarking&#34;](). Please star, watch and fork our repo for the active updates!

## 1. Overview

<!-- <div style="display: flex; justify-content: center;">
  <img src="https://github.com/usail-hkust/UUKG/blob/main/workflow.png" width="400">
  <img src="https://github.com/usail-hkust/UUKG/blob/main/UrbanKG.png" width="300">
</div> -->

In a rapidly evolving job market, skill demand forecasting is crucial as it enables policymakers and businesses to anticipate and adapt to changes, ensuring that workforce skills align with market needs, thereby enhancing productivity and competitiveness. Additionally, by identifying emerging skill requirements, it directs individuals towards relevant training and education opportunities, promoting continuous self-learning and development. However, the absence of comprehensive datasets presents a significant challenge, impeding research and the advancement of this field. To bridge this gap, we present **Job-SDF**, a dataset designed to train and benchmark job-skill demand forecasting models. Based on 10.35 million public job advertisements collected from major online recruitment platforms in China between 2021 and 2023, this dataset encompasses monthly recruitment demand for 2,324 types of skills across 521 companies. Our dataset uniquely enables evaluating skill demand forecasting models at various granularities, including occupation, company, and regional levels. We benchmark a range of models on this dataset, evaluating their performance in standard scenarios, in predictions focused on lower value ranges, and in the presence of structural breaks, providing new insights for further research.

## 2. Installation

Step 1: Create a python 3.8 environment and install dependencies:

```
conda create -n python3.8 Job-SDF
source activate Job-SDF
```

Step 2: Install library

```bash
pip install -r requirements.txt
```

## 3. Dataset 
Our dataset comprises five components for each granularity level: job skill demand sequences, job skill demand proportion sequences, ID mapping index, the indexes of skills with structural breaks, and skill co-occurrence graph, which can be found in dataset.


#### 3.1 Job-SDF Data

|  | Job AD|Region  | L1-Occupation | L2-Occupation  | Company   | Skill |
| ------- |  ------- | ------- | -------- | ------- | ------- | ------ |
| Size     | 10.35 million |7| 14       | 52 | 521 | 2335 |

##### 3.1.1 Guidance on data usage and processing

We store the processed files in the **'./dataset'** directory. 
The file information in each directory is as follows:

```
./demand    These are presented in tabular files, where each row represents a specific skill, and each column corresponds to a different time slice (month). Each cell within the table contains a numerical value that reflects the demand for the respective skill during that month.

./proportion    This component is also formatted in tabular files similar to the skill demand sequences. However, each cell in these tables displays a value between 0 and 1, representing the proportion of demand.

./structural_breaks_index     In the provided dataset, data concerning skills that have experienced structural breaks are organized in JSON format. Each granularity level is represented by a separate JSON file, which contains a list of indexes. These indexes correspond to the skills that have undergone structural breaks and can be directly mapped to the skill indexes in the skill demand sequences. The purpose of supplying this data is to facilitate research on the demand trends of skills that have exhibited structural breaks, enabling a detailed analysis of their demand dynamics over time.

./graph   This data is provided as a set of triples (skill ID\_1, skill ID\_2, frequency of co-occurrence), forming a collection that outlines the co-occurrence relationships between skills. Each triple indicates how frequently two skills are mentioned or required together within the job advertisements in the training data, serving as a prior knowledge graph to enhance predictive modeling by capturing relationships between skills.
```

## 4. How to Run

### 4.1 Traditional Methods
```bash
cd benchmark/traditional_method
python main.py [-h] [--data_name {r0, r1,...}]
      [--model {ARIMA, prophet}]
      [--mode {count, rate,...}]
```
### 4.2 Multi-variate time series forecasting
```bash
cd benchmark/multivariate_time_series
python run.py [-h] [--root_path {../../dataset/demand, ../../dataset/proportion}]
      [--data_path {r0.parquet, r1.parquet,...}]
      [--model {LSTM, CHGH, Autoformer,Crossformer,...}]
```
### 4.3 Pre-DyGAE
```bash
run data_process.ipynb
cd benchmark/predygae
sh scripts/stage1.sh {r0,r1,...}
sh scripts/stage2.sh {r0,r1,...} 24 36
sh scripts/stage3.sh {r0,r1,...} 24 36
```
### 4.4 Graph-based time series forecasting
```bash
cd benchmark/graph_method
python main.py [-h] [--data_name {r0, r1,...}]
      [--model {EvolveGCNH, EvolveGCNO}]
      [--mode {count, rate,...}]
```


## 5 Directory Structure

The expected structure of files is:

```
Job-SDF
 |-- benchmark
 |-- dataset  # Job-SDF_data
 |    |-- demand
 |    |-- graph
 |    |-- proportion
 |    |-- structural_breaks_index
 |-- figs
 |-- requirements.txt
 |-- README.md
```

<!-- ## 6 Citation

If you find our work is useful for your research, please consider citing:

```bash
@article{ning2024uukg,
  title={UUKG: unified urban knowledge graph dataset for urban spatiotemporal prediction},
  author={Ning, Yansong and Liu, Hao and Wang, Hao and Zeng, Zhenyu and Xiong, Hui},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
