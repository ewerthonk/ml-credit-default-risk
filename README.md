# Machine Learning - Home Credit Default Risk

*Predicting customer credit default with Machine Learning: a model garden exploration.*

# Project
## 📰 Intro
[Home Credit](https://www.homecredit.net) is a multinational company with a simple business model: lend money. Generally, to customers with little or no credit history.

One of the cornerstones of their business model is the **predictive power of the data they possess**. They are making some of this data available online aiming to unlock its full potential. In our case, we got a sample of this data to apply the data mining process on it with the intention to extract value.

## 🎯 Goal
The main goal is to use data to predict if an applicant will have trouble to pay their debt. This objective is cascaded into:

1. Creating a Machine Learning model that is good at distinguishing between classes: in our case, distinguishing a good applicant to a loan from a bad one.
2. This is also a Learning-oriented Project, I intend to **experiment on various aspects of Machine Learning** - and have some fun.

# 🗺  Methodology
![Project Overview](/references/project_overview.png)

# 🗄 Notebooks
- [1.0-framing-the-problem.ipynb](/notebooks/1.0-framing-the-problem.ipynb)
- [2.0-getting-the-data.ipynb](/notebooks/2.0-getting-the-data.ipynb)
- [3.0-eda-applications.ipynb](/notebooks/3.0-eda-applications.ipynb)
- [4.0-preparing-data-and-selecting-models.ipynb](/notebooks/4.0-preparing-data-and-selecting-models.ipynb)
- [5.0-fine-tunning-the-model.ipynb](/notebooks/5.0-fine-tunning-the-model.ipynb)

# 📦 Folder Structure

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate and useful data.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, figures, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as Markdown, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip list --format=freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, generate, and explore data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
