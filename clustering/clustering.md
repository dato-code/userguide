The
 [GraphLab clustering toolkit](https://dato.com/products/create/docs/graphlab.toolkits.clustering.html)
provides tools for unsupervised learning problems, where the aim is to
consolidate unlabeled data points into groups based on how similar the points
are to each other. The only clustering algorithms presently available are k-means and hierarchical k-means.
Our implementations of k-means and hierarchical k-means use
 [k-means++](http://en.wikipedia.org/wiki/K-means%2B%2B)
and
 [reservoir sampling](http://www.geeksforgeeks.org/reservoir-sampling/)
respectively to choose initial clusters.

##K-Means <a id="k-means"></a>

###What you'll need for this example
In this section, we explore a medical dataset from a June 2014 Kaggle
competition using k-means clustering. The dataset can be found at [MLSP 2014
Schizophrenia Classification Challenge](https://www.kaggle.com/c/mlsp-2014-mri),
which is the Kaggle page for the IEEE International Workshop on Machine Learning
for Signal Processing.

###Preparing the data
The original data consisted of two sets of features: functional network
connectivity (FNC) features and source-based morphometry (SBM) features. A
detailed description of these features is available at the Kaggle URL linked
above.

We incorporate both types of features into a single
[SFrame](https://dato.com/products/create/docs/generated/graphlab.SFrame.html).
The CSV containing FNC
features consists of 379 columns, where the first column is an integer ID for
each patient, and the remaining 378 columns are all floating point values. The
SBM data files share the same patient ID field, and have an additional 32
columns of features of type float. To combine the original data into a single
SFrame, we need to use the
[SFrame.join](https://dato.com/products/create/docs/generated/graphlab.SFrame.join.html)
method.

```python
# load FNC features
data_url = 'http://s3.amazonaws.com/dato-datasets/mlsp_2014/train_FNC.csv'
col_types = [int] + [float] * 378
fnc_sf = gl.SFrame.read_csv(data_url, column_type_hints=col_types)

# load SBM features
data_url = 'http://s3.amazonaws.com/dato-datasets/mlsp_2014/train_SBM.csv'
col_types = [int] + [float] * 32
sbm_sf = gl.SFrame.read_csv(data_url, column_type_hints=col_types)

# join all features on the Id column
dataset = fnc_sf.join(sbm_sf, on="Id")
```

###Tuning the model
For many clustering algorithms including k-means, you need to specify the number
of clusters the algorithm should create. Unfortunately, we rarely know the
correct number of clusters a priori. There is a simple heuristic that often
works quite well for estimating this parameter: $$k \approx \sqrt{n/2}$$ with
n as the number of rows in your dataset.

```python
from math import sqrt

n = len(fnc_sf)
k = int(sqrt(n / 2.0))

del dataset["Id"] # delete 'Id' column to exclude it from feature set

model = gl.kmeans.create(dataset, num_clusters=k)
```

###Performance improvements <a id="kmeans-performance"></a>
For large datasets, this training process can be very time-consuming. The
problem of partitioning *n* items into *k* clusters based on an item's distance
from the cluster mean is
 [NP-hard](http://en.wikipedia.org/wiki/NP-hard). There are a few tricks one might
employ to reduce the running time of the algorithm. One such improvement is to
reduce the total number of iterations required for convergence by making our
initial cluster assignments more accurate. This is precisely why our implementation
initializes cluster centers with the k-means++ algorithm. Here are a few other tips
for reducing the overall running time:

  - Cluster a sample of the original dataset.
  - Do some initial feature selection to reduce the feature space to the most
  discriminative features.

###Results
The model exposes two fields to help us understand how the algorithm has
clustered the data. The first is the `cluster_id` field, which gives us an
SFrame containing one row for each record in our input dataset. Each row of
`cluster_id` has a cluster assignment (an integer 0 to k, exclusive) and a
distance, which is the [euclidean
distance](http://en.wikipedia.org/wiki/Euclidean_distance) between the data
point and its cluster's center.

```python
model['cluster_id']
```

```no-highlight
    cluster_id    distance
0           3   6.591738
1           1   6.163163
2           3   7.194580
3           2   7.371710
4           4   7.303070
5           2   7.882903
6           4   6.130990
7           1   6.615896
8           3   8.299443
9           0   5.236333
10          1   9.129009
11          2   6.777277
12          4   6.796411
13          1   6.762669
14          3   6.384697
15          3   8.058596
16          1   6.773928
17          2   6.030693
18          2   7.900586
19          1   7.639997
20          2   7.767107
21          2   6.653062
22          4   8.572708
23          2   7.379239
24          2   6.338177
          ...        ...

[86 rows x 2 columns]
```

Equally interesting and useful for doing any post-hoc analysis is the
`cluster_info` field, which consists of another SFrame containing k rows (one
row per cluster). Each of these rows has a dimensionality equal to the input
dataset (ie. 378 FNC features + 32 SBM features + 1 ID), with values
representing the center of the corresponding cluster. Each row of the
`cluster_info` SFrame also contains the cluster ID number, the sum of distances
from all cluster members to the cluster cluster, and the number of data points
assigned to that cluster.

Here we print just the first 5 columns of the cluster info SFrame for the
purpose of illustration, followed by the cluster metadata.

```python
model['cluster_info'].print_rows(num_columns=5)
```
<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">FNC1</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">FNC2</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">FNC3</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">FNC4</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">FNC5</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.29645625</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.18360825</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.031482</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.22433625</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.27823125</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.214121916667</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.103074</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.0531845833333</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0384319166667</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.246585041667</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.176196448276</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.0303982758621</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.184082458621</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.176179551724</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.123517862069</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.150637533333</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.0209958666667</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.0950924666667</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.153809926667</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0386812133333</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.3431122</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.23126</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.0018778</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.04711628</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.100756</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.3168948</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0628264</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0947642</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.15517396</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.1654592</td>
    </tr>
</table>
[6 rows x 5 columns]<br/>
</div>


```python
model['cluster_info'][['cluster_id', '__within_distance__', '__size__']]
```
<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">cluster_id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">__within_distance__</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">__size__</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">55.0737148571</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">167.069080727</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">213.064455321</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">29</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">108.468702579</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">15</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.149361302</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.8373819666</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[6 rows x 3 columns]<br/>
</div>


##Hierarchical K-Means

This algorithm is an extensions on the [k-means](#k-means) algorithm described in the previous section. It involves recursively applying the k-means algorithm on a data set until some termination conditions are met. The result of hierarchical k-means is a tree-structure clustering in which each level of the tree is a more fine-grained partition of the data set than the previous level. For more details, see [this paper](http://www.andrew.cmu.edu/user/hgifford/projects/k_means.pdf).

###What you'll need for this example
In this section, we explore a dataset of vector space embeddings of words from 
the word2vec project at Google Research using hierarchical k-means. The dataset 
can be found at [word2vec](https://code.google.com/p/word2vec/), which is the 
Google Code page for the Google Research project. To repeat the experiments 
shown in this section, you will also need [gensim](https://radimrehurek.com/gensim/). 
Gensim's [word2vec implementation](https://radimrehurek.com/gensim/models/word2vec.html)
is quite easy to use and provides utilities for unpacking the word2vec binary files.

###Preparing the data
The data consists of real-valued vectors of 300 dimensions, each corresponding to a 
unique string observed in a 100 billion word data set. There is one vector per unique 
string. The vectors are acquired by running the word2vec algorithm on a corpus. The 
word2vec algorithm is a log-bilinear language model. Details can be found at the 
Google Code project page linked above. 

 > NOTE: It is important to keep in mind that the representations produced by log-
 bilinear models consist of latent variables with no interpretable meaning.

We use the gensim utility to unpack the vectors into Python lists of floats, then 
enter each word-list[float] pair its own row of an
[SFrame](https://dato.com/products/create/docs/generated/graphlab.SFrame.html). 
Below are some helper functions that we use to get the word vectors into an SFrame.

```python
from gensim.models import Word2Vec
from curses.ascii import isalpha, islower

import graphlab as gl

# filter out noisy words with some heuristics
def is_valid_word(word):

  # transform to ascii for filters below
  ascii_word = word.encode('ascii', errors='replace')

  # mark anything that is uppercase or non-alphabetic
  valid = [isalpha(c) and islower(c)
           for c in ascii_word]

  '''
  It may seem odd to filter out words with uppercase, but in our experience, they 
  were mostly proper nouns like usernames and people's first names, which is 
  quite noisy and too specific for our model
  '''

  # mark as invalid if any letters marked
  return all(valid)

# get vectors in form of list[float] from gensim model
def get_word_vectors(file_path, binary=True):

  # unpack word vectors from bin file
  # this may take a few minutes
  word2vec = Word2Vec.load_word2vec)format(file_path, binary=binary)

  # some funky business so I get lists instead of Python objects
  return {word: word2vec[word]
          for word in word2vec.vocab.keys()
          if is_valid_word(word)}

# Turn the dictionary from get_word_vectors into an SFrame
def get_word2vec_sf(word2vec):

  return gl.SFrame({
      'word': word2vec.keys(),
      'vector': word2vec.values()})

# To get the data set as an SFrame
dataset = get_word2vec_sf(
            get_word_vectors("your_word2vec_bin_file_path"))
```

###Performance improvements
Depending on the number of training examples, the dimensionality of the data, 
and the multi-threading capability of your hardware, training can take a long 
time. See [kmeans performance improvements](#kmeans-performance) and read the 
rest of this section.

There are a few different important (optional) hyperparameters for tuning the 
accuracy/speed trade-off of a hierarchical k-means model:

* _max_cluster_size_: if a cluster's member count goes below this number, the cluster is not further subclustered, so all leaf clusters have at most _max_cluster_size_ members
 * If you don't need clusters below a certain size, this parameter can be increased to reduce the number of levels that the model will attempt to complete. Note that results will be different for each random initialization and subset of the data.
* _max_depth_: once the cluster tree reaches this depth, the hierarchical k-means algorithm terminates
 * Decreasing this parameter will have similar results to increasing _max_cluster_size_, but it will behave exactly the same for all subsets of the data and random initializations.
* _max_changes_: if the number of changed cluster assignments between iterations for a level of the cluster hierarchy is <= this number, iteration on that level is terminated
 * If lower cluster quality is acceptable, this parameter can be increased to speed up the early termination of each level. Note that results will be different for each random initialization and subset of the data.
* _branch_factor_: each k-means subclustering produces _branch_factor_ new clusters
* _max_iterations_: once the number of iterations performed on a level of the cluster hierarchy reaches _max_iterations_, iteration for that level is terminated
 * Decreasing this parameter will have similar results to increasing _max_changes_, but it will behave exactly the same for all subsets of the data and random initializations.
* _cluster_scale_: this is a scaling constant for _max_cluster_size_; it is only used if _max_cluster_size_ is not set
 * If you would like to use our automatic _max_cluster_size_ selection, but you find that it is slightly too big or too small, you can set this parameter, and it will be used to scale the value of _max_cluster_size_.

Our implementation calculates default values for _max_changes_ based on the number of input data and _max_cluster_size_ based on the number of input data and the value of _branch_factor_.

```python
# Training the model
model = gl.toolkits.clustering.h_kmeans.create(
    dataset, 
    branch_factor=4, 
    max_iterations=50)
```
 > NOTE: We do not recommend using the raw representation of your data if it is very high-dimensional and/or sparse, like bag-of-words. One option transforming this data into 

###Raw results
Like flat k-means, hierarchical kmeans has the 'cluster_id' and 'cluster_info' 
fields, but with some additional columns. The 'cluster_id' field contains one 
additional column:

* _cluster_path_: this corresponds to the digit string of the leaf cluster to which a training example was assigned 

Keep in mind that the mapping from training example to cluster path is 
many-to-one. The 'cluster_info' field has a few additional columns:

* _cluster_path_: this is a string of digits that represents the path from the root cluster (the full dataset) to the current cluster
* _parent_id_: this is the unique integer id of this cluster's parent cluster
* _children_id_: this is a list of the unique integer ids of the subclusters of this id; an empty list indicates a leaf cluster
* _num_members_: the number of training examples assigned to this cluster

The [digit strings](#digit-strings) that represent cluster paths are explained in more detail later in the userguide.

Below is some example output for the _cluster_info_ field:

```python
model['cluster_info']

Columns:
  cluster_id    int
  parent_id     int
  children_id   list
  center        list
  num_members   int
  sum_squared_distance  float
  cluster_path  str

Rows: 881

Data:
+------------+-----------+------------------+-------------------------------+
| cluster_id | parent_id |   children_id    |             center            |
+------------+-----------+------------------+-------------------------------+
|     0      |     0     |   [1, 2, 3, 4]   | [[0.0, 0.0, 0.0, 0.0, 0.0,... |
|     1      |     0     |   [5, 6, 7, 8]   | [[-0.013871886330844602, 0... |
|     2      |     0     | [9, 10, 11, 12]  | [[0.005115983052361946, 0.... |
|     3      |     0     | [13, 14, 15, 16] | [[0.028774363911598766, 0.... |
|     4      |     0     | [17, 18, 19, 20] | [[-0.02957047659794855, 0.... |
|     5      |     1     | [21, 22, 23, 24] | [[0.0012539025527132685, 0... |
|     6      |     1     | [25, 26, 27, 28] | [[-0.014760119120969573, 0... |
|     7      |     1     | [29, 30, 31, 32] | [[-0.01852979825752836, 0.... |
|     8      |     1     | [33, 34, 35, 36] | [[-0.010061373369083259, -... |
|     9      |     2     | [37, 38, 39, 40] | [[-0.006255177732406638, -... |
+------------+-----------+------------------+-------------------------------+
+-------------+----------------------+--------------+
| num_members | sum_squared_distance | cluster_path |
+-------------+----------------------+--------------+
|    150000   |         0.0          |              |
|    19898    |    13991.2704743     |      0       |
|    57177    |    53272.9134985     |      1       |
|    49114    |    41576.1548369     |      2       |
|    23811    |    18298.9378922     |      3       |
|     2178    |    1237.53582602     |      00      |
|     3065    |    1754.11966104     |      01      |
|    10163    |    7355.40844499     |      02      |
|     4492    |    2446.05430312     |      03      |
|    14211    |    12698.7855099     |      10      |
+-------------+----------------------+--------------+
[881 rows x 7 columns]
```

Below is some example output for the _cluster_id_ field:

```python
model['cluster_id']

Columns:
  row_id        int
  cluster_id    int
  distance      float
  cluster_path  str

Rows: 150000

Data:
+--------+------------+----------------+--------------+
| row_id | cluster_id |    distance    | cluster_path |
+--------+------------+----------------+--------------+
|   0    |    637     | 0.820549982317 |    20000     |
|   1    |    697     | 0.891153838293 |    21000     |
|   2    |    385     | 0.778446810231 |    10000     |
|   3    |    275     | 0.836276477416 |     3022     |
|   4    |    837     | 0.802785920139 |    31010     |
|   5    |     25     | 0.647942776286 |     010      |
|   6    |     85     | 0.694921185829 |     0000     |
|   7    |    449     | 0.960522124393 |    11000     |
|   8    |    649     | 0.72643727292  |    20100     |
|   9    |    638     | 0.851073471217 |    20001     |
+--------+------------+----------------+--------------+
[150000 rows x 4 columns]
```

###Extra Notes

####Examining the clusters
The raw results returned by the model are concise and clean, but they are a 
little hard to interpret. If we would like to see the relationships between 
data points that the algorithm has exposed, we will need to do some 
transformations. 

We have provided a function `get_cluster_grouped_data` in the clustering module 
that transforms the data into a more intuitive grouping. Given the model, the 
data, and the name of an aggregation column from the original data, it will 
return an SFrame in which each row has a unique leaf cluster id and the list of 
elements from the aggregation column that were assigned to the leaf cluster 
with that id. The _with_cluster_info_ parameter will default to _True_, in 
which case the returned SFrame will also have the information from 
_cluster_info_ for each leaf cluster.

As you can see from the output below, you can now select a cluster by id and 
see the gruoped cluster members. This grouping can be helpful for checking 
cluster quality and for data exploration and visualization.

```python
get_cluster_grouped_data(model, data, 'word', with_cluster_info=False)

Columns:
  id    int
  cluster_id    int
  word_cluster  list

Rows: 661

Data:
+----+------------+-------------------------------+
| id | cluster_id |          word_cluster         |
+----+------------+-------------------------------+
| 0  |    118     | [bool, kaspersky, dng, xs,... |
| 1  |    435     | [multisourcing, multiservi... |
| 2  |    537     | [explusions, diplomat, pla... |
| 3  |    526     | [husbands, remarry, affect... |
| 4  |    511     | [clubrooms, heathland, str... |
| 5  |    363     | [wana, thas, feck, yankee,... |
| 6  |    431     | [cyberweapon, unpatched, k... |
| 7  |    738     | [sonority, fingerwork, arp... |
| 8  |    733     | [vocalization, personae, e... |
| 9  |    621     | [attacted, annnounced, unb... |
+----+------------+-------------------------------+
[661 rows x 3 columns]
```

####Digit Strings <a id="digit-strings"></a>
Each cluster in the tree can be uniquely represented by a pair of values: a 
unique identifier of its parent and a number from 0 until _k_ (i.e. 
_branch_factor_). Taking an inductive approach, we can start at the root of the 
cluster tree and uniquely represent each cluster in the tree as a string of 
digits from 0 until _k_ that represent clustering decisions at each level of the 
tree. 

At first this representation may seem silly and overly complicated compared to a 
unique integer id, but it can be very powerful. For example, let's say you want 
to use cluster membership of your training examples as features for several 
downstream models, and, either for performance or accuracy reasons, some models 
need more coarse-grained clustering of the data. Truncating the digit strings 
mapped to each example by _n_ characters give the clustering of the data at _n_ 
levels up from the leaves.

Because the length of the path from root to leaf is not necessarily the same for 
all leaf clusters, you must be cautious when truncating the strings. 
Fortunately, we provide and function `get_truncated_cluster_paths` to do this 
for you. 

```python
from graphlab.toolkits.clustering._util import get_truncated_cluster_paths

# set the number of levels to truncate from the tree
n = 2

trunc_clust_paths = get_truncated_cluster_paths(model, n)
```

If your end goal is to use the clusters as features in a down-stream model, this function can be used to tune the richness of the representation to accomodate your performance/accuracy trade-off needs. 

If you are more interested in data exploration and visualization, this function can be composed with `get_cluster_grouped_data` to view partitions of the data at various resolutions, possibly resulting in new insights into the relationships inherent in your data.
