---
title: "2.1 Prepare your dataset and upload it to Amazon S3"
weight: 1
---


## Open the following notebook to follow along

Notebook: `1_prepare_dataset.ipynb`

![](/images/setup/setup14.png)

{{% notice tip %}}
Feel free to follow along with the presenter on the stage
{{% /notice %}}

Let's start by importing necessary packages.
We'll use sagemaker and boto3 to access Amazon S3 and numpy and pandas to pre-process the dataset


```python
import sagemaker
import boto3
import pandas as pd
import numpy as np
```

Create a sagemaker session and get access to the current role

```python
sess = boto3.Session()
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
```

We'll use the default S3 bucket to save dataset, training jobs and artifacts.
You can use the sagemaker session to get the path to the default bucket. Use a custom prefix to save all the workshop artifacts.


```python
bucket = sagemaker_session.default_bucket()
prefix = "sagemaker_huggingface_workshop"
```

Print the role, bucket and region


```python
print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sagemaker_session.default_bucket()}")
print(f"sagemaker session region: {sagemaker_session.boto_region_name}")
```

## Preparing the dataset

Women's E-Commerce Clothing Reviews with 23,000 Customer Reviews and Ratings
https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

![](/images/training/training1.png)

Load dataset and extract only the reviews and ratings


```python
df = pd.read_csv('./data/Womens Clothing E-Commerce Reviews.csv')
df = df[['Review Text',	'Rating']]
df.columns = ['text', 'label']
df['label'] = df['label'] - 1

df = df.dropna()
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
unique, counts = np.unique(df['label'], return_counts=True)
plt.bar(unique, counts)

plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()
```
![](/images/training/training4.png)

Create a train, validate and test set


```python
train, validate, test = \
              np.split(df.sample(frac=1, random_state=42),
                       [int(.6*len(df)), int(.8*len(df))])

train.shape, validate.shape, test.shape
```


```python
train.head(10)
```
![](/images/training/training5.png)

Create separate files for train, validate and test


```python
train.to_csv(   './data/train.csv'   , index=False)
validate.to_csv('./data/validate.csv', index=False)
test.to_csv(    './data/test.csv'    , index=False)
```

Upload all 3 files to the default bucket in Amazon S3


```python
dataset_path = sagemaker_session.upload_data(path='data', key_prefix=f'{prefix}/data')
print(f'Dataset location: {dataset_path}')
```
