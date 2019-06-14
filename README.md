# text2topics
Insight AI project: topic modeling 

## Setup 

1. Clone repository to local and create a virtual environment, e.g. 

```
conda create -n text2topics python=3.7 
conda activate text2topics
cd text2topics 
pip install -r requirements.txt 
cd source
```

2. Run ```main.py``` to load data and initialize variables/path names. 

3. Set the number of topics and number of iterations you would like for the LDA model. e.g. 

``` 
N_TOPICS = 4
N_PASS = 20
``` 

4. Generate an LDA topic model by calling (exactly): 

```
ldamodel = LDA(clean_file,DTM_file,dict_file,model_file, N_TOPICS, N_PASS)
``` 





