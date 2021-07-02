# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from bert import Bert
from data_loader import full_generator
from model import get_model
import json


# prepare y
# open data
dataframe = pd.read_csv("input/occupation_seq.csv", encoding = "cp1257").fillna(0)
LPKdict=pd.read_excel("input/ISCO_relationship.xlsx").fillna(0)#, encoding = "cp1257"

LPKdict.lpk4=LPKdict.lpk4.astype(int)

# create hierarchical structure

lpkindexmap = {}

for i in range(1,5):
    # to split 4 digit code into additional 1, 2, 3 digits codes
    dataframe[f'LPK_{i}_kodas'] = dataframe['s_veik_prof'].astype(str).str[0:i].astype(int)
    # add names to professions
    dataframe = dataframe.join(LPKdict.set_index('lpk4'), on=f'LPK_{i}_kodas')
    dataframe.rename(columns={'lpk_pav': f'LPK_{i}_pav'}, inplace=True) 
    
    # dictionary for prediction
    lpkindexmap[i] = sorted(dataframe.set_index(f'LPK_{i}_kodas').to_dict()[f'LPK_{i}_pav'].items())
    
dataframe=dataframe.drop(['s_veik_prof'], axis=1)


with open('lpkindexmap.json', 'w') as f:
    json.dump(lpkindexmap, f)


#make sparse matrix as we need to save memory
mlb = MultiLabelBinarizer(sparse_output=True)

y = pd.DataFrame()
for i in range(1,5):
    d = (dataframe[['id', f'LPK_{i}_kodas']]
        .drop_duplicates()
        .groupby('id')[f'LPK_{i}_kodas']
        .apply(lambda g: g.values.tolist())
        .to_dict())
        
    df = pd.DataFrame(d.items(), columns=['id', f'lpk{i}'])
    d = pd.DataFrame(mlb.fit_transform(df[f'lpk{i}']), columns=[f'sparselpk{i}'])
    if y.empty:
        y = df
    y = y.join(pd.DataFrame(d, index=df.index))

# prepare x 

#open file 
edu = pd.read_csv("input/education_seq.csv", encoding = "cp1257").fillna("")
abstract=pd.read_csv("input/descriptions.csv", encoding = "cp1257").fillna("NA")


# make id for program
edu = edu.assign(ind=(edu['kodas']).astype('category').cat.codes)
edu=pd.merge(edu, abstract, on='kodas',  how='left').fillna("NA")

# prepare sparse dataset
edu_sparse=edu.groupby('id')['ind'].apply(lambda g: g.values.tolist()).to_dict()
edu_sparse=pd.DataFrame(edu_sparse.items(), columns=['id', 'programos_kodai'])

mlb = MultiLabelBinarizer(sparse_output=True)
mlb_transform=pd.DataFrame(mlb.fit_transform(edu_sparse['programos_kodai']), columns=['sparse_programoskodai'])

x_sp=edu_sparse.join(pd.DataFrame(mlb_transform, index=edu_sparse.index))

# prepare text dataset
x_txt=edu.groupby(['id'])['trans_abstract'].apply(','.join).reset_index()
dataframe=pd.merge(x_txt, x_sp, on='id',  how='left')

# join x and y variables for final dataset and drop cases where a person
# has education but has no occupation 
dataframe=pd.merge(dataframe, y, on='id',  how='left')

dataframe=dataframe.dropna()
print(dataframe.iloc[1])

#make BERT from text input

bert = Bert()

translated_abstracts_df = dataframe[['id', 'trans_abstract']]

embeddings_df = pd.DataFrame()

for embedding in bert.make_embeddings(translated_abstracts_df[['trans_abstract']]):
    embeddings_df.append(pd.DataFrame(embedding), ignore_index=True)

result = pd.concat([translated_abstracts_df, embeddings_df], axis=1, sort=False)

#final data for training
X=result.values[:,2:]
Y = dataframe.values[:,5:9]


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.2)
print(X_train.shape)
print(y_train.shape)


batch_size = 128
number_of_batches = int(np.ceil(X_train.shape[0]/batch_size))
val_batch_size = int(np.ceil(X_val.shape[0]/number_of_batches))
number_of_batches

#training
model = get_model(X_train, y_train)

history = model.fit_generator(
    generator=full_generator(X_train, y_train[:,1], y_train[:,2], y_train[:,3], y_train[:,0], batch_size, number_of_batches, True), 
    steps_per_epoch=number_of_batches,
    validation_data=full_generator(X_val, y_val[:,1], y_val[:,2], y_val[:,3], y_val[:,0], val_batch_size, number_of_batches, False),
    validation_steps=number_of_batches,
    epochs=30, verbose=1)

gen_data=full_generator(X_test, y_test[:,1], y_test[:,2], y_test[:,3], y_test[:,0],val_batch_size, number_of_batches, False, infinite=False)
res = model.evaluate_generator(gen_data)
print(res)

