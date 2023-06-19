# DiCOVA Challenge

This project was carried out with Elisa Ramírez as part of a course in Speech Technologies.

![Dicova](https://github.com/CesarCaramazana/SpeechTech_DiCOVA_Challenge/blob/main/Figures/DiCOVA.PNG)

## Project description

In this Dicova challenge the objective is to discriminate between COVID-affected and COVID-free patients using audio recordings of either coughs, breaths or speech. A dataset is provided, as well as a default feature extraction method to obtain the Mel Spectrogram (MFCC) of a wav file. The baseline is stablished using three classical Machine Learning approaches from the sci-kit learn library: a Logistic Regressor, a Random Forest classifier and a Multi Layer Perceptron. 

We propose two distinct methodologies: 1) a Convolutional Neural Network that uses the MFCC matrix as input and extracts the visual information contained in the frequecy domain, and 2) an ensemble method of light-weighted classifiers.

Performance was measured in terms of Area under the ROC curve (AUC) and compared against the baseline. We found out that the CNN approach, despite being the solution that achieved the highest rank in the original challenge, couldn't compete in our scenario due to the lack of data; and that the Ensemble method improved the stand-alone Random Forest and Logistic Regressor but still was outperformed by the Multilayer Perceptron. 



## Feature extraction
For the task we selected the dataset containing **recordings of coughs** from patients with and without Covid. The data were heavily imbalanced, with an approximate proportion of 90-10% samples of non-Covid and Covid respectively. 

The feature extraction was done independently for each of the two classifiers we propose:
  * For the CNN, we extracted the MFCC spectrogram with 15 coefficients and hop size of 512, without deltas. The audio was previously preprocessed first by eliminating silent segments and then by trimming or padding the signal to a length of 150000 samples with the aim of obtaining a fixed size MFCC matrix that could be batched during training. With the selected hyperparameters, these matrices were (341, 15). 
  * For the ensemble models we added the deltas to the MFCC, after applying the same preprocessing steps. 

An overview of the pipeline (for the MFCC without deltas) is shown in the figure below.

![Features](https://github.com/CesarCaramazana/SpeechTech_DiCOVA_Challenge/blob/main/Figures/feature_extraction.PNG)


## Models

### Convolutional Neural Network

We propose a Convolutional Neural Network with a rather shallow architecture, combining 3 layers of convolutions and 2 layers densly connected. In the convolutional layers we used batch normalization and ReLU activation functions; in the fully-connected we added dropout and a sigmoid activation to generate the output as the probability of class 1. 

![CNN](https://github.com/CesarCaramazana/SpeechTech_DiCOVA_Challenge/blob/main/Figures/CNN%20architecture.PNG)

The CNN model was trained several times to play with the different hyperparameters (learning rate, loss, batch size, etc). The reported model:

- **Loss function**: Focal Loss (Binary Cross Entropy).
- **Optimizer**: Adam.
- **Learning Rate**: 3e-5
- **Regularization**: L2-norm regularization.
- **Batch size**: 256.
- **Xavier weight initialization** [5].

Some techniques were applied to tackle some of the problematiques present in the dataset, such as **weighting each class** inversely proportional to its size (to account for class imbalance) --which was discarded in favor of the Focal Loss--, and the **stratification of the batches** so that the proportion of each class is consistent --discarded due to implementation errors--.

To fasten the proccess, training is carried out with GPU, which for the selected features and architecture took no more than 10 minutes at most.


### Ensemble method

In the experiments carried out below, we created an ensemble of a Random Forest classifier, a Logistic Regressor and a Multilayer Perceptron. The particular parameters of each approach were validated separately before creating the ensemble.

![Ensemble](https://github.com/CesarCaramazana/SpeechTech_DiCOVA_Challenge/blob/main/Figures/ensemble_method.PNG)


## Results

The summary of results are shown in the table below:

![Results](https://github.com/CesarCaramazana/SpeechTech_DiCOVA_Challenge/blob/main/Figures/results.PNG)


## Conclusions

By replicating the winning approach of the DiCOVA challenge we aimed to assess the validity of the solution rather than to benchmark performance with more novel architectures. As the performance we obtained was much worse --the winners obtained AUC=87.07, contrasting our AUC=59.14--, and after comparing closely the impact the differences may have had in the results, we identified that the dataset we used was several orders smaller than the original. In the Deep Learning paradigm, the data are more important than the models, and a CNN solution was not suitable in our scenario. 


Regarding the ensemble method, the results were improved with respect to the stand-alone methods by combining the ouputs of the individual classifiers with weights proportional to their performance. However, the variability (std) of the AUCs imply that the confidence intervals of the models overlap, so the improvement is not very significative.


Future lines of work involve gathering more data from the original dataset or applying data augmentation at the audio level, the selection of better performing models for the ensemble method and the fine-tuning of hyperparameters. 
