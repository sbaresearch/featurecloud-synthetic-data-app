# Synthetic Data FeatureCloud App 

## Description
A Synthetic Data Feature Cloud App, generating fully synthetic data with the synthetic data vault library. 

## Input 
- data.csv containing the original dataset (columns: features; rows: samples)

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
    data: data.csv
    sep: ","
  synthetic_data_vault:
    model: GaussianCopula
    number_of_rows: 1000
    synthetize_fields:
      - w-education
      - medexp
      - sol
      - h-occupation
      - method
      - w-religion
    categorical_fields:
      - w-education
      - medexp
      - sol
      - h-occupation
      - method
      - w-religion
  result:
    file: synthetic_data.csv
```

The config file allows to specify the following: 
- the model for generating synthetic data, the options include: GaussianCopula, CTGAN, TVAE, CopulaGAN. The default model is GaussianCopula.
- the number of rows to generate, if not specified the dafult value corresponds to the number of rows in the original dataset. 

Similarly, under the option synthesize_fields, the user can specify the columns to be synthetized and under the option categorical_fields, the user can specify which columns are categorical. The data types of the other fields are inferred automatically.

Furthermore, under the option anonymize_fields, the user can create fake data, indicating the types of the fields.
