import pandas as pd
import numpy as np
import tensorflow as tf


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

def group_by_game_session_and_sum(events, columns):
    """
    some columns are rolling counts by game session,
    take the max value of each game session then some for totals
    """
    series = pd.Series()
    for c in columns:
        series[c] = events.groupby('game_session')[c].max().sum()
    return series


def summarize_events(events):
    """
    takes a dataframe of events and returns a pd.Series with aggregate/summary values
    """
    events = events.sort_values('timestamp').reset_index()
    events = events.rename(columns={'game_session_x': 'game_session'}, errors='ignore')
    numeric_rows = ['event_count', 'game_time']
    aggregates = group_by_game_session_and_sum(events, numeric_rows)
    aggregates['game_time'] = aggregates['game_time'] / 1000
    aggregates['num_unique_days'] = num_unique_days(events['timestamp'])
    aggregates['elapsed_days'] = days_since_first_event(events['timestamp'])
    aggregates['last_world'] = events.tail(1)['world'].values[0]
    aggregates['last_game_session'] = events.tail(1)['game_session'].values[0]
    aggregates['type_counts'] = events.type.value_counts()
    aggregates['unique_game_sessions'] = events.game_session.unique().size
    return aggregates


def summarize_events_before_game_session(events, game_session):
    events_before = get_events_before_game_session(events, game_session)
    aggregates = summarize_events(events_before)
    labels = events[['title_y', 'num_correct',
       'num_incorrect', 'accuracy', 'accuracy_group']].iloc[0]
    row = aggregates.append(labels)
    return row


def split_features_and_labels(df):
    labels_df = df[['title_y', 'num_correct', 'num_incorrect',
                    'accuracy', 'accuracy_group', 'installation_id', 'game_session_y']].copy()
    feats_df = df.drop(
        ['title_y', 'num_correct', 'num_incorrect', 'installation_id', 'game_session_y', 'last_game_session',
         'accuracy', 'accuracy_group'], axis=1)
    return feats_df, labels_df


def basic_user_features_transform(train_data, train_labels=None):
    data = train_data[['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code',
                       'game_time', 'title', 'type', 'world']]
    if train_labels:
        train_w_labels = data.merge(train_labels, on='installation_id')
        groups = train_w_labels.groupby(['installation_id', 'game_session_y'])
    else:
        groups = data.groupby(['installation_id'])
    # game session y is index 1 of the group name
    # passing none to game session is for eval data, does not subset any of the data for each installation_id
    features = groups \
        .apply(lambda x: summarize_events_before_game_session(x, game_session=x.name[1] if len(x) == 2 else '')) \
        .reset_index()
    expanded_counts = features.type_counts.apply(pd.Series)
    # rename the type count columns
    expanded_counts.columns = [c.lower()+'_ct' for c in expanded_counts.columns]
    feats = pd.concat([features.drop(['type_counts'], axis=1), expanded_counts], axis=1)

    if train_labels:
        return split_features_and_labels(feats)
    else:
        return feats, None
