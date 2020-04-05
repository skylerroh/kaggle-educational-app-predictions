import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

###
# def get_last_event(group):
#     return group.sort_values('timestamp', ascending=False).iloc[0]
# last_events = test.groupby('installation_id').apply(get_last_event).event_id.value_counts()
# print(last_events.index) # ['7ad3efc6', '3bfd1a65', '90d848e0', '5b49460a', 'f56e0afc']
###
last_events_before_assessment = ['7ad3efc6', '3bfd1a65', '90d848e0', '5b49460a', 'f56e0afc']

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


def get_worst_score(group):
    return group.sort_values('accuracy_group').iloc[0]


def is_assessment(titles_series):
    def is_assessment_title(title):
        return "assessment" in title.lower()
    return titles_series.apply(lambda x: is_assessment_title(x))

def num_unique_days(timestamps):
    return pd.to_datetime(timestamps).apply(lambda x: x.date()).unique().size


def days_since_first_event(timestamps):
    dates = pd.to_datetime(timestamps).apply(lambda x: x.date())
    return (dates.max() - dates.min()).days


def get_events_before_game_session(events, game_session):
    game_session_index = events.index[(events.game_session == game_session) & \
                                       (events.event_id.isin(last_events_before_assessment))]
    if not game_session_index.empty:
        return events.loc[:game_session_index[-1]]
    else:
        return events

def group_by_game_session_and_sum(events, columns):
    """
    some columns are rolling counts by game session,
    take the max value of each game session then sum for totals
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
    numeric_rows = ['event_count', 'game_time']
    aggregates = group_by_game_session_and_sum(events, numeric_rows)
    aggregates['game_time'] = aggregates['game_time'] / 1000
    aggregates['num_unique_days'] = num_unique_days(events['timestamp'])
    aggregates['elapsed_days'] = days_since_first_event(events['timestamp'])
    aggregates['last_world'] = events.tail(1)['world'].values[0]
    aggregates['last_assessment'] = events[is_assessment(events['title'])].tail(1)['title'].values[0]
    aggregates['last_game_session'] = events.tail(1)['game_session'].values[0]
    aggregates['assessments_taken'] = events['title'][events.event_id.isin(last_events_before_assessment)].value_counts()
    aggregates['type_counts'] = events.type.value_counts()
    aggregates['event_code_counts'] = events.event_code.value_counts()
    aggregates['unique_game_sessions'] = events.game_session.unique().size
    return aggregates


def summarize_events_before_game_session(events, game_session):
    events = events.rename(columns={'game_session_x': 'game_session', 'title_x': 'title'}, errors='ignore')
    events_before = get_events_before_game_session(events, game_session)
    aggregates = summarize_events(events_before)
    try:
        labels = events[['title_y', 'num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']].iloc[0]
        row = aggregates.append(labels)
    except KeyError:
        row = aggregates
        # print("no label columns, just returning features")
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
    if train_labels is not None:
        train_w_labels = data.merge(train_labels, on='installation_id')
        groups = train_w_labels.groupby(['installation_id', 'game_session_y'])
    else:
        groups = data.groupby(['installation_id'])
    # game session y is index 1 of the group name
    # passing none to game session is for eval data, does not subset any of the data for each installation_id
    features = groups \
        .apply(lambda x: summarize_events_before_game_session(x, game_session=x.name[1] if len(x.name) == 2 else '')) \
        .reset_index()
    expanded_type_counts = features.type_counts.apply(pd.Series).fillna(0)
    # rename the type count columns
    expanded_type_counts.columns = [c.lower()+'_ct' for c in expanded_type_counts.columns]

    expanded_event_code_counts = features.event_code_counts.apply(pd.Series).fillna(0)
    # rename the event_code count columns
    expanded_event_code_counts.columns = ['event_{}_ct'.format(int(c)) for c in expanded_event_code_counts.columns]
    
    expanded_assessments_taken = features.assessments_taken.apply(pd.Series).fillna(0)
    
    feats = pd.concat([features.drop(['type_counts', 'event_code_counts', 'assessments_taken'], axis=1), expanded_type_counts, expanded_event_code_counts, expanded_assessments_taken], axis=1)

    if train_labels is not None:
        return split_features_and_labels(feats)
    else:
        return feats, None

def get_data_processing_pipe(feats, log_features, categorical_features):
    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = [c for c in feats.columns if c not in log_features+categorical_features]
    for f in log_features:
        non_zero_min = feats[f][feats[f]!=0].min()
        feats[f] = feats[f].replace([0], non_zero_min)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value=0)),
        ('scaler', StandardScaler())])

    numeric_log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value=1)),
        ('log_scale', FunctionTransformer(np.log)),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('num_log', numeric_log_transformer, log_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor

def main():
    train, test, train_labels, specs = read_data()
    # train_labels = train_labels.groupby(['installation_id', 'title']).apply(get_worst_score).reset_index(drop=True)
    feats, labels = basic_user_features_transform(train, train_labels)
    test_feats, _ = basic_user_features_transform(test)
    # Save checkpoint
    feats.to_csv('installation_features.csv', index=False)
    labels.to_csv('installation_labels.csv', index=False)
    test_feats.to_csv('test_feats.csv', index=False)
    

if __name__ == '__main__':
    main()