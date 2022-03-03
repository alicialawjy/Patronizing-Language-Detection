## Project Layout
Repository Structure
```
Patronizing-Language-Detection
├─ README.md
├─ Report Analysis
│  ├─ Part 1.ipynb
│  └─ part3 analysis
│     ├─ Part3.1.ipynb
│     └─ df_test3.1.csv
├─ archived
├─ data_augmentation.py
├─ data_sampling.ipynb
├─ datasets
│  ├─ df_test.csv
│  ├─ df_val.csv
│  ├─ task4_test.tsv
│  ├─ raw_data
│  │  ├─ dev_semeval_parids-labels.csv
│  │  ├─ df_preprocessing.csv
│  │  ├─ dontpatronizeme_categories.tsv
│  │  ├─ dontpatronizeme_pcl.tsv
│  │  └─ train_semeval_parids-labels.csv
│  └─ training_data
│     ├─ data-augmentation
│     │  ├─ df_updown_paraphrased.csv
│     │  ├─ df_updown_syn_and_para.csv
│     │  ├─ df_updown_syn_replace.csv
│     │  └─ minority-augmented
│     │     ├─ paraphrased.csv
│     │     ├─ syn_and_para.csv
│     │     └─ syn_replace.csv
│     ├─ data-sampling
│     │  ├─ df_downsample.csv
│     │  ├─ df_updownsample.csv
│     │  └─ df_upsample.csv
│     └─ df_train.csv
├─ dont_patronize_me.py
├─ final_model.py
└─ hugging-face-implementation.py
```

## Key Files and Directories
1. datasets directory </br>
Contains all the datasets used/ explored. Some notes:
    - raw_data directory: pure datasets provided (not modified)
    - training_data directory: 
        - df_train.csv: 70% of the raw dataset
        - data-sampling: see Section 2.1.1 of the report
        - data-augmentation: see Section 2.1.2 of the report. </br>
        Subfolder minority-augmented contains only the augmented data (not appended with the full train set). 
        Final training set is `/datasets/training_data/data-augmentation/df_updown_paraphrased.csv`
    - df_val.csv: internal validation set (15% of raw dataset).
    - df_test.csv: internal test set (15% of raw dataset).
    - task4_test.tsv: dataset used for the competition submission</br>
</br>

2. final_model.py </br>
Contains the final model used for the final submission. Achieves an F1 score = 0.5277

3. hugging-face-implementation.py </br>
A sentiment classifier model done using a hugging face RoBERTa model.

4. data_sampling.ipynb </br>
Jupyter notebook used to do the data split.

5. data_augmentation.py </br>
Python script used to carry out paraphrasing and synonym replacement to upsample minority dataset.

6. Report Analysis directory </br>
Jupyter notebooks used to run the data analysis mentioned in Section 1 and Section 3 of the report.

## How to run the code
1. (OPTIONAL) - To install the required packages at your current environment </br>
```
./install_environment.sh
```

3. To run the code </br>
```
python3 final_model.py
```

4. To run Jupyter Notebook
```
jupyter notebook
```

## Authors
* **Alicia Jiayun Law** - *ajl115@ic.ac.uk*
* **Chan Mun Fai** - *mc821@ic.ac.uk*
* **Chua Wei Jie** - *wc1021@ic.ac.uk*

