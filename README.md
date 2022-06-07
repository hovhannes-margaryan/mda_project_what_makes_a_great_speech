# MDA Project: what makes a great speech?

This project addresses the following question: what makes a great speech?
The following metrics are defined and hypothesised to influence the quality and importance of a speech: emotions, sentiment,
complexity, lexical richness, proportion of named entities, imagery, stop‐words, and mean sentence
length. First the extent to which these metrics influence speech quality and significance are measured
and importance and typical speeches are compared based on these measures. Second, four classifier
are trained to classify important and typical speeches and to confirm that previously defined measures
influence the importance of a speech. Random Forest with tuned hyper‐parameters achieves the
highest test accuracy: 76.60%.

# Project Structure 

The project includes the following classes: DataLoader, BasicDataset, 
SpeechDataset, Speech, preprocessors and 
Analysis of Speech notebook that conducts the analysis 
making use of the classes and their functionality. Dataloader is an iterable 
over the SpeechDataset. SpeechDataset is design to read the speeches from pdf 
format and return a Speech. The Speech class has all
the methods to compute and extract all the measures defined above. 
Preprocessors are can be used either for the BasicDataset to 
directly preprocess the data or they can be used in the 
methods of Speech for customization in case of non-default 
models are used as arguments. The functionalities
are implemented in a general way and thus
can be used for further analysis of
speeches. 

# Dataset

The dataset of important and typical speeches can be downloaded
from 
[here](https://drive.google.com/drive/folders/10EMbmBnxAhhGtiL6E64VztrJImRAWuXQ?usp=sharing). The speeches are scraped from 
[American Rhetoric](https://www.americanrhetoric.com/).
They should be placed in the dataset folder. 
The creation of the dataset with all the 
above-mentioned features in the Analysis
of Speech notebook can take three hours 
depending on the capacity of CPU. The creation of the dataset
makes use of Imagery words retrieved from 
[MRC Psycholinguistic Database](https://websites.psychology.uwa.edu.au/school/mrcdatabase/uwa_mrc.htm)
which can also
be downloaded from the link indicated above. 
The created dataset
with all the features can also be downloaded from the link
above to avoid waiting for the creation in the notebook. 
The dataset should be placed in the resources folder. 

# Structure of the Directory 

``` .
 ├── dataset
 │   └── important
 │   └── typical
 ├── resources
 │   └── dataset_all.csv
 │   └── visual_words.csv
 └── README.md 
 └── DataLoader.py 
 └── Dataset.py 
 └── Speech.py
 └── preprocessors.py
 └── utils.py
 └── analysis_of_speech.ipynb
 └── requirements.txt
```