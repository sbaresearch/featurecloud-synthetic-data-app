# Synthetic Data FeatureCloud App 

## Description
A Synthetic Data Feature Cloud App, generating synthetic data with the Synthetic Data Vault (SDV) library in Python [[1]](#Resources). 

## Input 
- data.txt containing the original dataset (columns: features; rows: samples)

### Output
- synthetic_data.csv containing the synthetic dataset generated with the given parameters.

#### Workflow
Can be combined with the following apps:
- Post: 
  - Preprocessing apps (e.g. Cross-validation, Normalization ...) 
  - Various analysis apps (e.g. Logistic Regression, Linear Regression ...)


#### Config 
Use the config file to set the parameters for the synthetic data generation. Upload it together with your data that will be synthesized. 

```
fc_synthetic_data: 
  local_dataset:
    data: data.txt
    sep: ","
  synthetic_data_vault:
    model: GaussianCopula
    number_of_rows: 300
    synthetize_fields:
      - age
      - workclass
      - education
      - education-num 
      - marital-status
      - occupation 
      - relationship
      - race
      - sex
      - capital-gain
      - capital-loss
      - hours-per-week
      - native-country
      - prediction
    categorical_fields:
      - workclass
      - education
      - education-num 
      - marital-status
      - relationship
      - race
      - sex
      - native-country
      - prediction
    anonymize_fields:
      - occupation : job 
  result:
    file: synthetic_data.csv
```

The config file allows to specify the following: 
- the model for generating synthetic data, the options include: GaussianCopula, CTGAN, TVAE, CopulaGAN. The default model is GaussianCopula.
- the number of rows to generate, if not specified the dafult value corresponds to the number of rows in the original dataset. 

Similarly, under the option synthesize_fields, the user can specify the columns to be synthetized and under the option categorical_fields, the user can specify which columns are categorical. The data types of the other fields are inferred automatically.

Furthermore, under the option anonymize_fields, the user can create fake data for fields labeled as Personally Identifiable Information with the same statistical properties. To do this, as shown in the configuration example indicate the name of the field and the category. For checking the possible categories, we refer the reader to [Python Faker Documentation](https://faker.readthedocs.io/en/master/providers.html).

For more information, we refer the reader to the [SDV Documentation](https://sdv.dev/SDV/developer_guides/sdv/metadata.html).

## Resources

[1]. N. Patki, R. Wedge, and K. Veeramachaneni, [The Synthetic Data Vault.](https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf), IEEE International Conference on Data Science and Advanced Analytics (DSAA), 2016,pp. 399-410, doi: 10.1109/DSAA.2016.49.
