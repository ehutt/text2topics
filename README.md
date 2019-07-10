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







