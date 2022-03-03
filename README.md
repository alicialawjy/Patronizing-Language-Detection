## Project Layout
Repository Structure
```
Patronizing-Language-Detection
├─ README.md
├─ Semeval_2022_Task_4_Demo.ipynb
├─ archived
├─ classification_report.py
├─ classification_report2.py
├─ data-augmentation.py
├─ data_augmentation.py
├─ datasets
│  ├─ data_aug
│  ├─ intermediate_data
│  │  ├─ paraphrased.csv
│  │  ├─ syn_and_para.csv
│  │  └─ syn_replace.csv
│  ├─ official_test_data
│  │  └─ task4_test.tsv
│  ├─ raw_data
│  │  ├─ dev_semeval_parids-labels.csv
│  │  ├─ df_preprocessing.csv
│  │  ├─ dontpatronizeme_categories.tsv
│  │  ├─ dontpatronizeme_pcl.tsv
│  │  └─ train_semeval_parids-labels.csv
│  ├─ test_data
│  │  └─ df_test.csv
│  ├─ training_data
│  │  ├─ augmented_data
│  │  │  ├─ df_updown_paraphrased.csv
│  │  │  ├─ df_updown_sample.csv
│  │  │  ├─ df_updown_sample_03.csv
│  │  │  ├─ df_updown_syn_and_para.csv
│  │  │  ├─ df_updown_syn_replace.csv
│  │  │  └─ df_upsample_dup_syn.csv
│  │  ├─ balanced_data
│  │  │  ├─ df_downsample.csv
│  │  │  ├─ df_upsample_dup_syn.csv
│  │  │  └─ df_upsample_simple_dup.csv
│  │  ├─ basic data
│  │  │  ├─ df_downsample.csv
│  │  │  └─ df_upsample_simple_dup.csv
│  │  └─ df_train.csv
│  └─ validation_data
│     └─ df_val.csv
├─ dont_patronize_me.py
├─ evaluation.py
├─ final_model.py
├─ hugging-face-implementation.py
├─ shell_script.sh
└─ view_result.sh
```

The main directories/files of interest include:
1. datasets directory </br>
Contains all the datasets used/ explored. 
    - raw_data directory: pure datasets (not modified)
    - training_data directory: 70% of the dataset
        - xxx_data: see Section 2.1.1 of the report
        - augmented_data: see Section 2.1.2 of the report
    - validation_data directory: 15% of dataset
    - test_data directory: 15% of dataset
    - official_test_data: dataset used for the competition submission</br>
</br>
2. final_model.py </br>
Contains the final model used for the final submission.

3. hugging-face-implementation.py </br>
A sentiment classifier model done using a hugging face RoBERTa model.

4. data_preprocessing.ipynb </br>
Jupyter notebook used to do the data split

5. data-augmentation.py </br>
Python script used to carry out paraphrasing and synonym replacement to upsample minority dataset.

6. data_analysis.ipynb </br>
Jupyter notebook used to do data/ result analysis.

## Authors
* **Alicia Jiayun Law** - *ajl115@ic.ac.uk*
* **Chan Mun Fai** - *mc821@ic.ac.uk*
* **Chua Wei Jie** - *wc1021@ic.ac.uk*