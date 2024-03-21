假设我们有一小组用户和几部电影，我们通过用户对电影的评分来训练一个简单的推荐模型。为了简化，我们使用Python和Pandas来构建一个简单的例子。  
首先，我会创造一些示例数据，然后演示如何使用这些数据来推荐电影给特定的用户。  
创造示例数据：我们将创建一个简单的用户-电影评分矩阵，其中包含几个用户对几部电影的评分。 
基于最简单的推荐逻辑：我们将使用最基本的推荐逻辑，比如推荐给用户他们还未观看的评分最高的电影。

```python
# 创建示例电影评分数据
import pandas as pd
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'User 1': [5, 4, None, 2, 1],
    'User 2': [None, 3, 4, 2, 1],
    'User 3': [2, None, 5, 4, None],
    'User 4': [None, 2, 3, None, 4]
}

# 将数据转换为DataFrame
ratings = pd.DataFrame(data).set_index('Movie')

# 展示数据
ratings
```
<img src="https://github.com/ALANSUNENET/Recommender_system/assets/126316591/bdcf216a-75f8-486f-951d-fe77784cdc88" width="70%" />

为了实现基于用户和基于电影的协同过滤，我们会先保持原有的数据，然后展示两种不同的方法：
  - 基于用户的协同过滤 (User-Based Collaborative Filtering)：这种方法找出与目标用户有相似喜好的其他用户，然后推荐那些用户喜欢的电影给目标用户。
  - 基于电影的协同过滤 (Item-Based Collaborative Filtering)：这种方法则是基于电影之间的相似性来进行推荐。如果一个用户喜欢了某部电影，系统会找出与这部电影相似的其他电影来推荐。
我们将使用之前创建的用户-电影评分数据。首先，我们需要处理数据，为两种方法准备相应的相似度矩阵。

## 步骤 1: 准备相似度矩阵
对于用户相似度矩阵，我们将计算用户间的相似度。
对于电影相似度矩阵，我们将计算电影间的相似度。

这通常通过皮尔逊相关系数、余弦相似度等方法来计算。

```python
# 填充缺失值为0（对于协同过滤计算来说，NaN不利于计算）
ratings_filled = ratings.fillna(0)

# 计算用户相似度矩阵
user_similarity = pd.DataFrame(
    cosine_similarity(ratings_filled.T),
    index=ratings.columns,
    columns=ratings.columns
)

# 计算电影相似度矩阵
movie_similarity = pd.DataFrame(
    cosine_similarity(ratings_filled),
    index=ratings.index,
    columns=ratings.index
)

user_similarity, movie_similarity
```


<img src="https://github.com/ALANSUNENET/Recommender_system/assets/126316591/75050be1-0217-47a8-80df-c42c2c1000cc" width="70%" />

<img src="https://github.com/ALANSUNENET/Recommender_system/assets/126316591/e98a0e30-4104-4fdb-9880-c41b76cf6d71" width="70%" />

基于用户的协同过滤推荐：我们将找出每个用户最相似的其他用户，并基于这些相似用户的评分来推荐电影。

基于电影的协同过滤推荐：我们将为用户推荐与他们已评分电影最相似的其他电影。


## 步骤 2: 实现基于用户的协同过滤推荐
基于用户的协同过滤（User-Based Collaborative Filtering）是一种推荐系统方法，它通过寻找目标用户的“邻居”（即与目标用户有相似品味的其他用户）来进行推荐。具体来说，该方法执行以下步骤：

计算用户间的相似度：首先，我们计算所有用户之间的相似度。在我们的例子中，我们使用了余弦相似度来计算用户间的相似度。

找到相似的用户：然后，我们找出与目标用户最相似的用户。这些用户的喜好被认为与目标用户类似。

生成推荐：基于这些相似用户对电影的评分，我们可以预测目标用户对尚未观看的电影的潜在喜好，并据此生成推荐。

## 步骤 3: 实现基于电影的协同过滤推荐
基于电影的协同过滤（Item-Based Collaborative Filtering）关注电影（或一般来说，项目）之间的相似性。它为用户推荐那些与用户已知喜好的电影相似的电影。具体步骤如下：

计算电影间的相似度：我们计算所有电影之间的相似度。这也是通过余弦相似度完成的。

找到相似的电影：对于目标用户已评分的每部电影，我们找出与之最相似的电影。

生成推荐：基于这些相似电影及其与目标电影的相似度，我们预测目标用户可能对这些相似电影的喜好，并据此生成推荐。
