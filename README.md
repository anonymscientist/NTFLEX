# Complex Query Answering over Temporal Knowledge Graphs with Literals

This repository contains the code for the paper "Complex Query Answering over Temporal Knowledge Graphs with Literals". The results of our experiments are presented in the `experiments` folder.

## Installation

Follow these steps to set up the NTFLEX enviroment on your computer:

### Step 1: Clone the NTFLEX repository

First, clone the NTFLEX repository from GitHub using the following command:

```shell
git clone https://github.com/annonymdeveloper/NTFLEX.git
cd NTFLEX
```

### Step 2: Create the Conda Environment

```shell
conda create --name ntflex python=3.9.18 && conda activate ntflex
```

### Step 3: Install Dependencies

To install the required packages, run this command in the NTFLEX folder:

```shell
pip install -r requirements.txt
```

### Creation of the Dataset (Optional)

The dataset we used was created from a Wikidata dataset by García-Durán et al. (https://github.com/mniepert/mmkb/tree/master/TemporalKGs/wikidata). This data can be found in the folder `data/original`. We used the Wikidata Rest-API to modify this dataset with following script:

```shell
python create_dataset.py --access_token=["Enter valid Wikidata-Access-Token"]
```

This dataset stores facts as quintuples with (`subject, predicate/attribute, object/attribute value, since, until`). Since our framework uses quadruples we need to preprocess the data. To preprocess it and split the quadruples into train/test/valid splits run the following script:

```shell
python preprocess.py
```

The preprocessed data already exists in the `data/WIKI` folder, so you can skip this step if you want to use the data we used in our experiments.

## Recreate NTFLEX results

To recreate the NTFLEX results published in our paper run following command:

```shell
python train_NTFLEX.py
```

## Recreate TFLEX results

Follow these steps to recreate the TFLEX results presented in our paper.

First, head to the TFLEX folder:

```shell
cd TFLEX
```

To train the TFLEX model on the WIKI dataset, run the following command:

```shell
python train_TCQE_TFLEX.py --dataset "WIKI"
```

## Recreate TransEA results

Follow these steps to recreate the TransEA results presented in our paper

### Step 1: Create the Conda Enviroment

```shell
conda create --name transea python=3.7.16 && conda activate transea
```

### Step 2: Install Tensorflow

```shell
pip install tensorflow==1.15.0
```

### Step 3: Compile 

```shell
bash makeEA.sh
```

### Step 4: Train

To train models based on random initialization:

1. Change class Config in transEA.py and set testFlag to False

		class Config(object):
	
			def __init__(self):
				...
				self.testFlag = False
				self.loadFromData = False
				...

2.
```shell
python transEA.py
```

### Step 5: Test

To test your models:

1. Change class Config in transEA.py and set testFlag to True
	
		class Config(object):

			def __init__(self):
				...
				self.testFlag = True
				self.loadFromData = True
				...

2.
```shell
python transEA.py
```

