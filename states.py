from FeatureCloud.app.engine.app import app_state, AppState, Role, LogLevel
from FeatureCloud.app.engine.app import State as op_state
import pandas as pd
from sdv import SDV
from sdv import Metadata
from sdv.tabular.copulagan import CopulaGAN
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.ctgan import CTGAN, TVAE
import bios

INPUT_DIR = 'mnt/input'
OUTPUT_DIR = 'mnt/output'

INITIAL_STATE = 'initial'
WRITE_STATE = 'WriteSyntheticData'
TERMINAL_STATE = 'terminal'


# App name
app_name="fc_synthetic_data"

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state(INITIAL_STATE)
class InitialState(AppState):

    def register(self):
        self.register_transition(WRITE_STATE)  

    def run(self):
        print('Loading config...')
        self.read_config()
        print('Loading data...') 
        df=self.read_data()
        synthetic_data =  self.generate_synthetic_data(df)
        output_file= f"{OUTPUT_DIR}/{self.config['result']['file']}"
        self.store('output_file', output_file)
        self.store('original_data', df)
        self.store('synthetic_data', synthetic_data)
        return 'WriteSyntheticData'
    
    def read_config(self):
        self.config = bios.read(f'{INPUT_DIR}/config.yml')[app_name]  

    def read_data(self):
        file_name = f"{INPUT_DIR}/{self.config['local_dataset']['data']}"
        format= file_name.split('.')[-1]
        if format=='csv' or format == 'txt':
            df = pd.read_csv(file_name, sep=self.config['local_dataset']['sep'])
        else:
            self.app.log(f"The file format {format} is not supported", LogLevel.ERROR)
            self.update(state=op_state.ERROR)
        return df

    def generate_synthetic_data(self,data):
        metadata=self.tables_config(data)
        model_str = self.config['synthetic_data_vault'].get('model', False)
        if model_str: 
            print(metadata.get_table_meta('table'))
            sdv = self.models_configuration(model_str, metadata)
            sdv.fit(data)
        else:
            sdv = SDV()
            sdv.fit(metadata,data)
        number_of_rows = self.config['synthetic_data_vault'].get('number_of_rows',False)
        if number_of_rows:
            synthetic_data = sdv.sample(num_rows=number_of_rows)
        else:
            synthetic_data = sdv.sample_all()
        return synthetic_data

    def tables_config(self,data):
        metadata = Metadata()
        categorical_attributes = self.config['synthetic_data_vault'].get('categorical_fields',False)
        fields_synthetization  = self.config['synthetic_data_vault'].get('synthetize_fields', False)
        fields_anonymization   = self.config['synthetic_data_vault'].get('anonymize_fields',False)
        if not fields_synthetization:
            fields_synthetization= list(data.columns)
        field_types={}
        if categorical_attributes:
            for category in  categorical_attributes:
                field_types[category]= { 'type' : 'categorical'}
        metadata.add_table(
            name='table',
            data=data,
            fields=fields_synthetization,
            fields_metadata=field_types  
        )
        if fields_anonymization:
            for field,category in fields_anonymization[0].items():
                if field in field_types:
                    field_types[field]['pii']      = True 
                    field_types[field]['pii_category'] = category
                else:
                    field_types[field]= {'pii':True, 'pii_category': category}
        return metadata

    def models_configuration(self, model_str, metadata):
        table_metadata=metadata.get_table_meta('table')
        if (model_str=='GaussianCopula'):
            return GaussianCopula(table_metadata=table_metadata)
        elif(model_str=='CTGAN'):
            return CTGAN(table_metadata=table_metadata)
        elif(model_str=='TVAE'):
            return TVAE(table_metadata=table_metadata)
        elif (model_str=='CopulaGAN'):
            return CopulaGAN(table_metadata=table_metadata)
        else:
            self.app.log(f"The model {model_str} is not supported", LogLevel.ERROR)
            self.update(state=op_state.ERROR)

@app_state(name=WRITE_STATE, role=Role.BOTH)
class WriteResults(AppState):
    def register(self):
        self.register_transition(TERMINAL_STATE, Role.BOTH)

    def run(self):
        output_file=self.load('output_file')
        orig_df=self.load('original_data')
        samples=self.load('synthetic_data')
        syn_df = samples
        syn_df.to_csv(output_file, index=False)
        print('Synthetic Data Created Succesfully!!!')
        print(syn_df.head())
        return TERMINAL_STATE