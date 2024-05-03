import pandas as pd

# 加载数据
movies = pd.read_csv('/Users/wecky/Documents/bysj/movielens/ml-1m_movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
ratings = pd.read_csv('/Users/wecky/Documents/bysj/movielens/ml-1m_ratings.dat', sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='ISO-8859-1')

# 选择前500个用户和前500个电影
selected_users = ratings['UserID'].unique()[:500]
selected_movies = movies['MovieID'].unique()[:500]

# 过滤出这些用户和电影的评分数据
selected_ratings = ratings[ratings['UserID'].isin(selected_users) & ratings['MovieID'].isin(selected_movies)]

# 过滤出这些电影的信息
selected_movies_info = movies[movies['MovieID'].isin(selected_movies)]

# 保存到新的文件
with open('/Users/wecky/Documents/bysj/movielens/selected_ratings.dat', 'w') as f:
    for index, row in selected_ratings.iterrows():
        f.write('::'.join(row.astype(str).values) + '\n')

with open('/Users/wecky/Documents/bysj/movielens/selected_movies.dat', 'w') as f:
    for index, row in selected_movies_info.iterrows():
        f.write('::'.join(row.astype(str).values) + '\n')
