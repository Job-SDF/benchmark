<div align="center">
    <p> 
    <img src="https://github.com/Job-SDF/benchmark/blob/main/figs/title.pdf" width="440px">
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

<!-- ## 3. Dataset

We opensource two large-scale Urban Knowledge Graph (UrbanKG) datasets in New York and Chicago compatible with five Urban SpatioTemporal Prediction (USTP) tasks. As the original dataset is quite large, we have included example data, data processing code, and model code to assist researchers in understanding our work. The complete data sources can be found on [Google Drive](https://drive.google.com/drive/folders/1egTmnKRzTQuyW_hsbFURUonGC-bJmBHW?usp=sharing).

We provide very detailed explanation for our data and pre-processing module in both [UrbanKG construction](https://github.com/usail-hkust/UUKG/tree/main/UrbanKG_data) and [USTP dataset construction](https://github.com/usail-hkust/UUKG/tree/main/USTP_data). The above dataset construction scheme is highly reusable, one can prepare their own urban data and use our code to build their personalized UrbanKG and USTP dataset easily.

#### 3.1 UrbanKG Data

| Dataset | Entity  | Relation | Triplet | Train   | Valid  | Test   |
| ------- | ------- | -------- | ------- | ------- | ------ | ------ |
| NYC     | 236,287 | 13       | 930,240 | 837,216 | 46,512 | 46,512 |
| CHI     | 140,602 | 13       | 564,400 | 507,960 | 28,220 | 28,220 |

##### 3.1.1 Guidance on data usage and processing

We store the original unprocessed files in the **'./Meta_data'** directory. To preprocess, align, and filter these files, we utilize either the **`preprocess_meta_data_nyc.py`** or **`preprocess_meta_data_chi.py`** script. The processed data is then saved in the **'./Processed_data'** directory. Finally, we execute the **`construct_UrbanKG_NYC.py`** or **`construct_UrbanKG_CHI.py`** script to obtain the constructed urban knowledge graphs, which are stored in the **'./UrbanKG'** directory.

We divide the training set, verification set and test set by **`train_val_test_ent2id_rel2id.py`**

The file information in each directory is as follows:

```
./Meta_data    Raw data set: administrative division data, POI and road network data
./Processed_data   Aligned datasets: administrative region entity, POI entity, road network entity
./UrbanKG    City fact triples obtained from 8 entities and 13 relationships
```

The following types of atomic files are defined:

| filename            | content                                 | example              |
| ------------------- | --------------------------------------- | -------------------- |
| entity2id_XXX.txt   | entity_name, entity_id                  | Road/106710 11       |
| relation2id_XXX.txt | relation_name, relation_id              | BNB 2                |
| train               | entity_id, relation_id, entity_id       | 196632  12 85987     |
| valid               | entity_id, relation_id, entity_id       | 43982   10 233474    |
| test                | entity_id, relation_id, entity_id       | 167134  6  75149     |
| triplet.txt         | entity_id, relation_id, entity_id       | 48034   12 168303    |
| UrbanKG_XXX.txt     | entity_name, relation_name, entity_name | POI/663 PLA Area/230 |

##### 3.1.2 To create your urban knowledge graph dataset

Our urban knowledge graph construction scheme is highly reusable. You can prepare your urban data following either the file format in **'./Meta_data'** or **'./Processed_data'**, and then run scripts **`construct_UrbanKG_XXX.py`** to build your personalized urban knowledge graph. This flexibility allows you to adapt the construction process to various cities and datasets easily.

##### 3.1.3 Visualization

<img src="https://github.com/usail-hkust/UUKG/blob/main/UrbanKG_data/visualization.png" width="440px">

We offer comprehensive visualization solutions for all types of urban knowledge. By leveraging the powerful visualization capabilities of **Folium**, we provide an intuitive understanding of the urban entities and relationships encoded in the constructed urban knowledge map. This allows users to interact with and explore the urban knowledge graph in a user-friendly manner, facilitating better insights and analysis of the urban data.

You can run **`UrbanKG_visulization_XXX.py`** to get the overall visualization of urban entities like borough, area, POI and road segment. You can also develop other visualization function according to your preferences.

#### 3.2 USTP Data

| Type    | USTP flow prediction                | USTP event prediction |
| ------- | ----------------------------------- | --------------------- |
| Dataset | taxi, bike, human Mobility          | crime, 311 service    |
| Sensor  | region-level, road-level, POI-level | region-level          |

##### 3.2.1 Guidance on data usage and processing

We store the original unprocessed files in the **'./Meta_data'** directory. To preprocess, align, and filter these files, we utilize either the **`preprocess_meta_data_nyc.py`** or **`preprocess_meta_data_chi.py`** script. The processed data is then saved in the **'./Processed_data'** directory.

Finally, we execute the **`construct_USTP_Pointflow_XXX.py`** script to obtain the spatiotemporal flow prediction dataset and derive **`construct_USTP_Event_XXX.py`** script to obtain the constructed urban event prediction dataset.

We storage them in the  **'./USTP'** directory with the special format mentioned in [here](https://github.com/usail-hkust/UUKG/blob/main/USTP_Model/readme.md).

The file information in each directory is as follows:

```
./Meta_data    Raw data set: taxi, bike, crime and 311 service event data.
./Processed_data   Aligned datasets: taxi, bike, human, crime and 311 service spatiotemporal dataset which are aligned with area, road and POI.
./USTP    The reformatted USTP dataset is now ready for use with downstream USTP models. 
```

The following types of atomic files are defined:

| filename    | content                                                             | example                                     |
| ----------- | ------------------------------------------------------------------- | ------------------------------------------- |
| xxx.geo     | Store geographic entity attribute information.                      | geo_id, type, coordinates                   |
| xxx.rel     | Store the relationship information between entities, such as areas. | rel_id, type, origin_id, destination_id     |
| xxx.dyna    | Store traffic condition information.                                | dyna_id, type, time, entity_id, location_id |
| config.json | Used to supplement the description of the above table information.  |                                             |

we explain the above four atomic files as follows:

**xxx.geo**: An element in the Geo table consists of the following four parts:

**geo_id, type, coordinates.**

```
geo_id: The primary key uniquely determines a geo entity.
type: The type of geo. These three values are consistent with the points, lines and planes in Geojson.
coordinates: Array or nested array composed of float type. Describe the location information of the geo entity, using the coordinates format of Geojson.
```

**xxx.rel**: An element in the Rel table consists of the following four parts:

**rel_id, type, origin_id, destination_id.**

```
rel_id: The primary key uniquely determines the relationship between entities.
type: The type of rel. Range in [usr, geo], which indicates whether the relationship is based on geo or usr.
origin_id: The ID of the origin of the relationship, which is either in the Geo table or in the Usr table.
destination_id: The ID of the destination of the relationship, which is one of the Geo table or the Usr table.
```

**xxx.dyna**: An element in the Dyna table consists of the following five parts:

**dyna_id, type, time, entity_id(multiple columns**.

```
dyna_id: The primary key uniquely determines a record in the Dyna table.
type: The type of dyna. There are two values: label (for event-based task) and state (for traffic state prediction task).
time: Time information, using the date and time combination notation in ISO-8601 standard, such as: 2020-12-07T02:59:46Z.
entity_id: Describe which entity the record is based on, which is the ID of geo or usr.
```

**xxx.config**: The config file is used to supplement the information describing the above five tables themselves. It is stored in `json` format and consists of six keys: `geo`, `usr`, `rel`, `dyna`, `ext`, and `info`.

##### 3.2.2 To create your USTP dataset

Our urban spatiotemporal prediction dataset construction scheme is highly reusable. You can prepare your urban downstream task data following either the file format in **'./Meta_data'** or **'./Processed_data'**, and then run scripts **`construct_USTP_Pointflow_XXX.py`** or **`construct_USTP_Event_XXX.py`** to build your personalized USTP dataset. This flexibility allows you to adapt the construction process to various cities and datasets easily.

##### 3.2.3 Visualization

<img src="https://github.com/usail-hkust/UUKG/blob/main/USTP_data/bike_start_end.png" width="650px">

We offer spatial and temporal visualization implement for all types of USTP dataset. By leveraging the powerful visualization capabilities of **Folium**, we provide an intuitive understanding of different USTP tasks.

You can run **`visualize_USTP.py`** to get the overall spatial and temporal distribution of USTP dataset. You can also develop other visualization function according to your preferences.

## 4. How to Run

#### 4.1 Structure-aware UrbanKG Embedding

To train and evaluate a UrbanKG embedding model for the link prediction task, use the run.py script:

```bash
python ./UrbanKG_Embedding_Model/run.py 
			 [-h] [--dataset {NYC, CHI}]
              [--model {TransE, RotH, ...}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

```

**How to get the embedding?**

We build the index between entities and learned embeddings and storage the index file in **./data/entity_idx_embedding.csv**. To obtain the learned UrbanKG embedding, run **`get_embedding.py`**.

#### 4.2 Knowledge-enhanced Urban SpatioTemporal Prediction

To train and evaluate a USTP model for the link prediction task, use the run.py script:

```bash
python ./USTP_Model/run.py --task traffic_state_pred --model STGCN --dataset NYCTaxi20200406
```

This script will run the STGCN model on the NYCTaxi20200406 dataset for traffic state prediction task under the default configuration.

**How to fuse UrbanKG embedding?**

To fuse UrbanKG embedding, we directly concatenate the embedding with USTP feature for input. You can mannualy modify it in the **`./data/dataset/traffic_state_dataset.py`**.

The **"readme.md"** file in [USTP_Model](https://github.com/usail-hkust/UUKG/tree/main/USTP_Model) and [UrbanKG_Embedding_Model](https://github.com/usail-hkust/UUKG/tree/main/UrbanKG_Embedding_Model) provide more details about models.

## 5 Directory Structure

The expected structure of files is:

```
Job-SDF
 |-- dataset  # Job-SDF_data
 |    |-- demand
 |-- baselines
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
``` --> -->
