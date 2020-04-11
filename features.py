import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from joblib import Parallel, delayed
import multiprocessing

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name, group) for name, group in dfGrouped)
    return pd.DataFrame(retLst)

###
# def get_last_event(group):
#     return group.sort_values('timestamp', ascending=False).iloc[0]
# last_events = test.groupby('installation_id').apply(get_last_event).event_id.value_counts()
# print(last_events.index) # ['7ad3efc6', '3bfd1a65', '90d848e0', '5b49460a', 'f56e0afc']
###
last_event_before_assessment = {'Cauldron Filler (Assessment)': '90d848e0',
                                'Cart Balancer (Assessment)': '7ad3efc6',
                                'Mushroom Sorter (Assessment)': '3bfd1a65',
                                'Bird Measurer (Assessment)': 'f56e0afc',
                                'Chest Sorter (Assessment)': '5b49460a'}

media_seq = pd.read_csv('media_sequence.csv')
clips_seq = media_seq[media_seq.type=='Clip']
clip_times = dict(zip(clips_seq.title, clips_seq.duration))


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

## FIX
def get_events_before_game_session(events, game_session, assessment_title):
    if not (game_session or assessment_title):
        return events
    else:
        assessment_event = last_event_before_assessment.get(assessment_title)
        game_session_index = events.index[(events.game_session == game_session) & \
                                           (events.event_id.isin([assessment_event] if assessment_event else last_event_before_assessment.values()))]
        return events.loc[:game_session_index[-1]]


def group_by_game_session_and_sum(events, columns):
    """
    some columns are rolling counts by game session,
    take the max value of each game session then sum for totals
    """
    series = pd.Series(dtype=int)
    for c in columns:
        # set beginning values for each type to 0
        for stype in ['activity', 'game', 'assessment', 'clip']:
            series[stype+'_'+c] = 0
        series['total_'+c] = 0 
        
        # get session type and total values and add to running total
        for session_id, session in events.groupby('game_session'):
            session_type = session['type'].iloc[0].lower()
            session_value = session[c].max() / 1000.0 if c=='game_time' else session[c].max()
            series[session_type+'_'+c] += session_value
            series['total_'+c] += session_value
        if c=='game_time':
            series = series.drop(labels='clip_'+c)
    return series


def summarize_events(events):
    """
    takes a dataframe of events and returns a pd.Series with aggregate/summary values
    """
    events['ts'] = pd.to_datetime(events.timestamp)
    events = events.sort_values('ts').reset_index()
    events['ts_diff'] = -events.ts.diff(-1).dt.total_seconds()
    numeric_rows = ['event_count', 'game_time']
    aggregates = group_by_game_session_and_sum(events, numeric_rows)
    aggregates['num_unique_days'] = num_unique_days(events['timestamp'])
    aggregates['elapsed_days'] = days_since_first_event(events['timestamp'])
    aggregates['last_world'] = events.tail(1)['world'].values[0]
    aggregates['last_assessment'] = events[is_assessment(events['title'])].tail(1)['title'].values[0]
    aggregates['assessments_taken'] = events['title'][events.event_id.isin(last_event_before_assessment.values())].value_counts()
    aggregates['type_counts'] = events[['game_session', 'type']].drop_duplicates()['type'].value_counts()
    aggregates['title_counts'] = events[['game_session', 'title']].drop_duplicates()['title'].value_counts()
    aggregates['event_code_counts'] = events.event_code.value_counts()
    aggregates['event_id_counts'] = events.event_id.value_counts()
    aggregates['unique_game_sessions'] = events.game_session.unique().size
    return aggregates


def summarize_events_before_game_session(name, events):
    if not isinstance(name, (list,tuple)) or len(name)==1:
        game_session=None
        assessment=None
    else:
        installation_id, game_session, assessment = name
    
    events = events.rename(columns={'game_session_x': 'game_session', 'title_x': 'title'}, errors='ignore')
    events_before = get_events_before_game_session(events, game_session, assessment)
    aggregates = summarize_events(events_before)
    try:
        labels = events[['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']].iloc[0] \
            .append(pd.Series(name, index=['installation_id', 'game_session_y', 'title_y']))
        row = aggregates.append(labels)
    except KeyError:
        row = aggregates
        # print("no label columns, just returning features")
    return row


def expand_count_features(features):
    print('**expanding event type count features**')
    expanded_type_counts = features.type_counts.apply(pd.Series).fillna(0)
    # rename the type count columns
    expanded_type_counts.columns = [c.lower()+'_ct' for c in expanded_type_counts.columns]
    
    print('**expanding title count features**')
    expanded_title_counts = features.title_counts.apply(pd.Series).fillna(0)
    # rename the type count columns
    expanded_title_counts.columns = [c.lower().replace(' ', '_')+'_ct' for c in expanded_title_counts.columns]
    

    print('**expanding event code count features**')
    expanded_event_code_counts = features.event_code_counts.apply(pd.Series).fillna(0)
    # rename the event_code count columns
    expanded_event_code_counts.columns = ['event_{}_ct'.format(int(c)) for c in expanded_event_code_counts.columns]
    # non_zero_event_code_counts 
    for ec in expanded_event_code_counts.columns:
        expanded_event_code_counts['non_zero_'+ec] = (expanded_event_code_counts[ec] > 0).astype(int)
    
    print('**expanding event id count features**')
    expanded_event_id_counts = features.event_id_counts.apply(pd.Series).fillna(0)
    # rename the event_id count columns
    expanded_event_id_counts.columns = ['eid_{}_ct'.format(c) for c in expanded_event_id_counts.columns]
    
    expanded_assessments_taken = features.assessments_taken.apply(pd.Series).fillna(0)
    
    feats = pd.concat([features.drop(['type_counts', 'title_counts', 'event_code_counts', 'event_id_counts', 'assessments_taken'], axis=1), expanded_type_counts, expanded_title_counts, expanded_event_code_counts, expanded_event_id_counts, expanded_assessments_taken], axis=1)
    return feats


def split_features_and_labels(df):
    labels_df = df[['title_y', 'num_correct', 'num_incorrect',
                    'accuracy', 'accuracy_group', 'installation_id', 'game_session_y']].copy()
    feats_df = df.drop(
        ['title_y', 'num_correct', 'num_incorrect', 'game_session_y',
         'accuracy', 'accuracy_group'], axis=1)
    return feats_df, labels_df


def basic_user_features_transform(train_data, train_labels=None):
    data = train_data[['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code',
                       'game_time', 'title', 'type', 'world']]
    if train_labels is not None:
        train_w_labels = data.merge(train_labels, on='installation_id')
        groups = train_w_labels.groupby(['installation_id', 'game_session_y', 'title_y'])
    else:
        groups = data.groupby(['installation_id'])
    # game session y is index 1 of the group name
    # passing none to game session is for eval data, does not subset any of the data for each installation_id
    print('**getting user features before each training assessment**')
    features = applyParallel(groups,
                             lambda name, group: summarize_events_before_game_session(name, group))
    
    expanded_features = expand_count_features(features)
    

    if train_labels is not None:
        return split_features_and_labels(expanded_features)
    else:
        return expanded_features, None

def get_data_processing_pipe(feats, log_features, categorical_features):
    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = [c for c in feats.columns if c not in log_features+categorical_features+['installation_id']]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value=0, strategy='constant')),
        ('scaler', StandardScaler())])

    numeric_log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(fill_value=0, strategy='constant')),
        ('log_scale', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        remainder='drop',
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('num_log', numeric_log_transformer, log_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor

def main():
    train, test, train_labels, specs = read_data()
    # train_labels = train_labels.groupby(['installation_id', 'title']).apply(get_worst_score).reset_index(drop=True)
    print('**transforming training data**')
    feats, labels = basic_user_features_transform(train, train_labels)
    print('**transforming test data**')
    test_feats, _ = basic_user_features_transform(test)
    # Save checkpoint
    print('**saving csvs**')
    feats.to_csv('installation_features_v2.csv', index=False)
    labels.to_csv('installation_labels_v2.csv', index=False)
    test_feats.to_csv('test_feats_v2.csv', index=False)
    

if __name__ == '__main__':
    main()