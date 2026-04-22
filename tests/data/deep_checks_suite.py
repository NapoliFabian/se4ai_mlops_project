import pandas as pd
from sklearn.model_selection import train_test_split
from deepchecks.nlp.suites import data_integrity, train_test_validation
from deepchecks.tabular.checks.data_integrity import ColumnsInfo
from deepchecks.tabular.checks.data_integrity import DataDuplicates
from deepchecks.tabular import Dataset
from deepchecks.nlp import TextData

file_path = 'data/raw/dataset.csv'

df = pd.read_csv(file_path).fillna('')

# solve out of memory
if len(df) > 20000:
    df = df.sample(n=10000, random_state=42)
df['full_text'] = (df['title'] + " " + df['text']).str.slice(0, 20000)

# 2. Split del dataset in Train e Test
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

df = 0
ds_train_2 = Dataset(df_train, label="label", cat_features=[])
ds_test_2 = Dataset(df_test, label="label", cat_features=[])

ds_train = TextData(
    df_train['full_text'].tolist(),
    label=df_train['label'].tolist(),
    task_type='text_classification'
)

ds_test = TextData(
    df_test['full_text'].tolist(),
    label=df_test['label'].tolist(),
    task_type='text_classification'
)

check = ColumnsInfo()
check.run(ds_train_2)
check.run(ds_test_2)


DataDuplicates().run(ds_train_2)
DataDuplicates().run(ds_test_2)



#Integrity Suite
integrity_result = data_integrity().run(ds_train)
integrity_result.save_as_html('reports/nlp_data_integrity.html')

#Train-Test Validation Suite

validation_suite = train_test_validation()
validation_result = validation_suite.run(ds_train, ds_test)
validation_result.save_as_html('reports/nlp_train_test_validation.html')

print("Deepcheck pipeline completed, file saved in 'nlp_data_integrity.html' and 'nlp_train_test_validation.html'")