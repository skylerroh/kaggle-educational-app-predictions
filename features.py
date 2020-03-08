import pandas as pd
import numpy as np 
# import pyspark
# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
import tensorflow as tf

# sc = SparkContext.getOrCreate('local')
# spark = SparkSession(sc)

def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    return train, test, train_labels, specs

def num_unique_days(timestamps):
	return pd.to_datetime(timestamps).apply(lambda x: x.date()).unique().size

def days_since_first_event(timestamps):
	dates = pd.to_datetime(timestamps).apply(lambda x: x.date())
	return dates.max() - dates.min()

def get_events_before_game_session(events, game_session):
	game_session_index = events.index[(events.game_session_x == game_session)]
	if not game_session_index.empty:
		return events.loc[:game_session_index[-1]]
	else:
		return events

def summarize_events(events):
	"""
	takes a dataframe of events and returns a pd.Series with aggregate/summary values
	"""
	events = events.sort_values('timestamp').reset_index()
	events = events.rename(columns={'game_session_x': 'game_session'}, errors='ignore')
	numeric_rows = ['event_count', 'game_time']
	aggregates = events[numeric_rows].sum()
	aggregates['num_unique_days'] = num_unique_days(events['timestamp'])
	aggregates['elapsed_days'] = days_since_first_event(events['timestamp'])
	aggregates['last_world'] = events.tail(1)['world'].values[0]
	aggregates['type_counts'] = events.type.value_counts()
	aggregates['unique_game_sessions'] = events.game_session.unique().size
	return aggregates

def summarize_events_before_game_session(events, game_session):
	# game_session = events['game_session_y'].min()
	events_before = get_events_before_game_session(events, game_session)
	aggregates = summarize_events(events_before)
	labels = events[['title_y', 'num_correct',
       'num_incorrect', 'accuracy', 'accuracy_group']].iloc[0]
	row = aggregates.append(labels)
	return row


def get_basic_user_features(train_data, train_labels):
	train_w_labels = train_data[['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code', 'game_time', 'title',
           'type', 'world']].merge(train_labels, on='installation_id')
	groups = train_w_labels.groupby(['installation_id', 'game_session_y'])
	features = groups.apply(lambda x: summarize_events_before_game_session(x, game_session=x.name[1])).reset_index()
	expanded_counts = features.type_counts.apply(pd.Series)
	return pd.concat([features.drop(['type_counts'], axis=1), expanded_counts], axis=1)
