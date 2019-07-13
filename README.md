# text2topics

Insight Artificial Intelligence, Session 2019B

## Background

As an AI fellow at Insight Data Science, I completed a consulting project with
[Ping](https://www.timebyping.com/), a company that offers automated 
timekeeping software to law firms. 

I built this topic modeling package to help them more quickly and accurately 
analyze large amounts of unstructured text in the form of emails, documents, 
and memos that lawyers write each day. 

This program leverages the power of natural language processing (NLP) and 
unsupervised learning to build a topic model that accurately identifies 
and represents the key topics in unstructured text. 

Specifically, it is designed to take a set of documents and generate a 
topic model specific to the legal domain. The resulting topic model can 
provide insights into the nature of an existing corpus and can also be 
used to perform inference-- that is, to identify the key topics in an 
unseen document. 

![Alt text](../images/intro.png)


Additional Info: [Slides](http://bit.ly/text2topics_slides)

## Setup 

1. Git clone this repository to local and create a virtual environment: 

```
conda create -n text2topics python=3.7 
conda activate text2topics
cd text2topics 
```

2. Install required packages:
```
pip install -r requirements.txt 
```

3. If first time running, type in the command line: 

```
python -m spacy download en
spacy download en_core_web_lg
```

## Usage 

### Example 1

The following command will: 
    1. load raw text data 
    2. clean and process it 
    3. iterate over different versions of the model
    
Use the resulting plot and the elbow method to identify the best number of topics. 

```
cd source 
python run.py \
    --raw_file ../data/raw_docs.pkl \
    --clean_path ../data/ \
    --results_path ../results/ \
    --clean_from_raw True \
    --iterate True \
    --use_lda False \
```

### Example 2

The following command will: 
    1. load already cleaned data 
    2. perform embedding + clustering procedure with a user provided number of topics
    3. generate word clouds for each of the identified topics, saving to results folder
    
Use the resulting model and word clouds to explore the nature of the topics and documents. 
```
cd source 
python run.py \
    --raw_file ../data/raw_docs.pkl \
    --clean_path ../data/ \
    --results_path ../results/ \
    --clean_from_raw False \
    --iterate False \
    --use_lda False \
```
### Example 3

To perform the above examples using LDA rather than the embeddings + clustering approach, pass the argument: 
```
    --use_lda True \
```


## How It Works 













