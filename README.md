# kaggle-educational-app-predictions

This W207 Project is based off of the kaggle competition: https://www.kaggle.com/c/data-science-bowl-2019
The data is from PBS KIDS Measure Up! app from which the task is to use application event data to predict scores on in-game assessments and create an algorithm that will lead to better-designed games and improved learning outcomes.

The data can be downloaded here https://www.kaggle.com/c/data-science-bowl-2019/data or via the kaggle api `kaggle competitions download -c data-science-bowl-2019`

The outcomes in this competition are grouped into 4 groups (labeled accuracy_group in the data):

> 3: the assessment was solved on the first attempt.  
> 2: the assessment was solved on the second attempt.  
> 1: the assessment was solved after 3 or more attempts.  
> 0: the assessment was never solved.   

Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two outcomes. The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix O is constructed, such that Oi,j corresponds to the number of installation_ids i (actual) that received a predicted value j. An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted values:  
  
![w_{i,j}= \frac{(i-j)^2}{(N-1)^2}](https://latex.codecogs.com/gif.latex?w_{i,j}=&space;\frac{(i-j)^2}{(N-1)^2})
 
An N-by-N histogram matrix of expected outcomes, E, is calculated assuming that there is no correlation between values.  This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that E and O have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as: 
  
![K = 1 - (\sum_{i,j} w_{i,j}O_{i,j} / \sum_{i,j} w_{i,j}E_{i,j})](https://latex.codecogs.com/gif.latex?K&space;=&space;1&space;-&space;(\sum_{i,j}&space;w_{i,j}O_{i,j}&space;/&space;\sum_{i,j}&space;w_{i,j}E_{i,j}))

## Feature Engineering
This dataset requires substantial feature engineering as the training and test datasets given are event logs for each `installation_id` of the PBS KIDS Measure Up! app. The training dataset contains 11,341,042 events for 17,690 installation_id/assessments combinations. 
In order to predict the assessment accuracy group these events need to be transformed into feature vectors for each of the 17,690 training labels. The following steps are done to extract such vectors:  

1. Join labels to events by `installation_id`
2. Group events by `installation_id`, label `game_session` and `assessment_title`
3. Filter events to only those that occured before the assessment
4. Get various features from the group of events
	- number of unique days in the history
	- number of unique game sessions
	- elapsed days since first event
	- which assessments have already been taken
	- `event_code` counts
	- `event_id` counts
	- media `title` counts
	- total `event_count` for each activity type
	- total `game_time` for each activity type
	- the last `world` of the user
	- does the user skip video clips (video lessons)
	- how much of the videos does the user watch on avg
	- ...  
	
Many of the features in the dataset are zero heavy with a lognormal distribution for values > 0 as shown below. To address this, the preprocessing pipeline will log normalize many of the features:

- event_code counts
- event_id counts
- total event counts for each activity type
- game_times for each activity type

## Data Preprocess Pipe
All of the features written to `installation_features_v2.csv` are raw integers / floats / strings for each installation, assessment training point. Before passing the feature vector to our models, a series of transformations are performed on the various columns and returned a vector of floats.

 - arg: `log_features`: list[column_names] -> np.log1p transformation -> standard scalar
 - arg: `categorical_features`: list[column_names] -> one_hot_encoder
 - `numerical_features`: columns not in log_features + categorical_features -> standard scalar

## Final Model
### User Based KFold Splits
CV scores on regular Kfold splits yielding greater quadratic weighted kappa scores compared to final private data set scores on Kaggle. This is partially due to data leakages when different assessments for the same `installation_id` are split amongst the train and eval sets. Since the test set is on entirely new users. We should simulate similar K eval sets with `installation_id`s that are distinct from the training set. This is done using the UserCV subclass of kfold that assigns a set percentage of `installation_id`s to each fold rather than random rows. This will result in slightly higher variance in end train/test size but users remain in distinct groups.

### Ensemble Ordinal Classifier
1. XGBoostRegressor
2. TFKerasSequential MLP
3. OptimizeThresholds for quadratic weighted kappa using linear solver
