# Author: https://github.com/witheunjin/NCF-TF_ejlee.git

import pandas as pd
import numpy as np


def load_dataset():
    """
    데이터 로드 함수

    uids: train user
    mids: train movie
    users: 전체 user
    movies: 전체 movie
    df_train
    df_test
    """

    # 데이터 로드
    df = pd.read_csv('ratings.csv', header=None)
    df = df.drop(df.columns[3], axis=1)
    df.columns = ['userId', 'movieId', 'rating']
    df = df.dropna()
    df = df.loc[df.rating != 0]

    # user 샘플링
    sample_num = 100000
    unique_user_lst = list(np.unique(df['userId']))  # 358857명
    sample_user_idx = np.random.choice(len(unique_user_lst), sample_num, replace=True)
    sample_user_lst = [unique_user_lst[idx] for idx in sample_user_idx]
    df = df[df['userId'].isin(sample_user_lst)]
    df = df.reset_index(drop=True)

    # 1명 이상의 artist 데이터가 있는 user 만 사용
    df_count = df.groupby(['userId']).count()
    df['count'] = df.groupby('userId')['userId'].transform('count')
    df = df[df['count'] > 1]

    # user, movie 아이디 부여
    df['user_id'] = df['userId'].astype("category").cat.codes
    df['movie_id'] = df['movieId'].astype("category").cat.codes

    # lookup 테이블 생성
    movie_lookup = df[['movie_id', 'movieId']].drop_duplicates()
    movie_lookup['movie_id'] = movie_lookup.movie_id.astype(str)

    # train, test 데이터 생성
    df = df[['user_id', 'movie_id', 'rating']]
    df_train, df_test = train_test_split(df)

    # 전체 user, movie 리스트 생성
    users = list(np.sort(df.user_id.unique()))
    movies = list(np.sort(df.movie_id.unique()))

    # train user, movie 리스트 생성
    rows = df_train['user_id'].astype(int)
    cols = df_train['movie_id'].astype(int)
    values = list(df_train.rating)

    uids = np.array(rows.tolist())
    mids = np.array(cols.tolist())

    # 각 user 마다 negative movie 생성
    df_neg = get_negatives(uids, mids, movies, df_test)

    df_train.to_csv('ml-100K.train.rating', sep='\t')
    df_test.to_csv('ml-100K.test.rating', sep='\t')
    df_neg.to_csv('ml-100K.test.negative', sep='\t')

    return uids, mids, df_train, df_test, df_neg, users, movies, movie_lookup


def get_negatives(uids, mids, movies, df_test):
    """
    negative movie 리스트 생성함수
    """
    negativeList = []
    test_u = df_test['user_id'].values.tolist()
    test_i = df_test['movie_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))  # test (user, movie)세트
    zipped = set(zip(uids, mids))  # train (user, movie)세트

    for (u, i) in test_ratings:

        negatives = []
        negatives.append((u, i))
        for t in range(100):
            j = np.random.randint(len(movies))  # neg_movie j 1개 샘플링
            while (u, j) in zipped:  # j가 train에 있으면 다시뽑고, 없으면 선택
                j = np.random.randint(len(movies))
            negatives.append(j)
        negativeList.append(negatives)  # [(0,pos), neg, neg, ...]

    df_neg = pd.DataFrame(negativeList)

    return df_neg


def train_test_split(df):
    """
    train, test 나누는 함수
    """
    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    # df_test
    # user_id와 holdout_movie_id(user가 플레이한 아이템 중 1개)뽑기
    df_test = df_test.groupby(['user_id']).first()
    df_test['user_id'] = df_test.index
    df_test = df_test[['user_id', 'movie_id', 'rating']]
    df_test = df_test.reset_index(drop=True)

    # df_train
    # user_id 리스트에 make_first()적용
    mask = df.groupby(['user_id'])['user_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test


def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0  # [0,1,1,....]

    return result


if __name__ == '__main__':
    load_dataset()
