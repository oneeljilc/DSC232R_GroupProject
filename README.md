# DSC232R: Group Project - Spring 2025
Created by: Jillian O'Neel (A69037969) <br>
This project contains code developed for the Spring 2025 DSC232R Group Project.
## 1. Introduction
The [“100 Million+ Steam Reviews”](https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data) dataset contains over 113 million reviews submitted by users of the PC games digital distribution platform, Steam. This platform allows users to purchase, download, and play PC video games, as well as engage in community activities such as leaving reviews/ratings. This dataset contains complete review text as well as rich metadata about the reviews and their authors. This dataset is combined with the ["Steam Games"](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) dataset which contains additional metadata about the games themselves (such as name, release date, price, genre, etc.). This dataset was chosen for the DSC232R Group Project because the team members are avid users of Steam and have a personal connection and interest in the subject matter. Additionally, while various lectures in the Data Science degree program have touched on applications related to consumer products, there has not been an assignment or project yet that addresses this topic. In general, analysis of product reviews and product performance data can be used to create models that support product development, marketing, and overall customer insights. For example, a good predictive machine learning model can dynamically price products based on demand, evaluate the effectiveness of an ad campaign, and provide tailored recommendations, among other uses.
## 2. Methods
### 2.a. Data Importing
The “100 Million+ Steam Reviews” dataset is provided as a .zip file (containing a .csv file) and the "Steam Games" dataset is provided as a .csv file. For ease of use, files were downloaded from Kaggle and uploaded to Google Cloud Storage, where they are accessible via a public url. The following code downloads the files and saves them as spark dataframes. See the attached notebook for the environment set-up and other specific implementation details. <br>
```
url1 = ("https://storage.googleapis.com/dsc232r-group-project-data/steam-reviews.zip")
!wget "{url1}"
!unzip steam-reviews.zip -d /home/joneel/joneel/Group_Project/raw_data/steam-reviews && rm steam-reviews.zip
url2 = ("https://storage.googleapis.com/dsc232r-group-project-data/games.csv")
!wget "{url2}"
!mv games.csv /home/joneel/joneel/Group_Project/raw_data

reviews_df = sc.read.csv("reviews.csv", header=True, inferSchema=True)
games_df = sc.read.csv("games.csv", header=True, inferSchema=True)
```
### 2.b. PreProcessing
#### 2.b.i. Dataset #1: Steam Reviews
The following post-processing was completed on the `reviews_df` spark dataframe. In general, this involved dropping nulls/duplicates and casting to the correct datatypes per the Kaggle documentation.
```
# Dropping columns not useful for analysis and nulls/duplicates
reviews_df = reviews_df.drop("hidden_in_steam_china", "steam_china_location")
reviews_df_processed = reviews_df.filter(reviews_df.language == 'english')
reviews_df_processed = reviews_df_processed.na.drop(subset=["recommendationid", "appid", "author_steamid", "review"])
reviews_df_processed = reviews_df_processed.dropDuplicates(subset=["recommendationid"])

# Casting to correct datatypes
reviews_df_processed = reviews_df_processed.withColumn("author_num_games_owned", f.col("author_num_games_owned").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("author_num_reviews", f.col("author_num_reviews").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("author_playtime_forever", f.col("author_playtime_forever").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("author_playtime_last_two_weeks", f.col("author_playtime_last_two_weeks").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("author_playtime_at_review", f.col("author_playtime_at_review").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("author_last_played", f.from_unixtime(f.col("author_last_played")).cast("timestamp"))
reviews_df_processed = reviews_df_processed.withColumn("timestamp_created", f.from_unixtime(f.col("timestamp_created")).cast("timestamp"))
reviews_df_processed = reviews_df_processed.withColumn("timestamp_updated", f.from_unixtime(f.col("timestamp_updated")).cast("timestamp"))
reviews_df_processed = reviews_df_processed.withColumn("voted_up", f.col("voted_up").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("votes_up", f.col("votes_up").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("votes_funny", f.col("votes_funny").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("weighted_vote_score", f.col("weighted_vote_score").cast("double"))
reviews_df_processed = reviews_df_processed.withColumn("comment_count", f.col("comment_count").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("steam_purchase", f.col("steam_purchase").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("received_for_free", f.col("received_for_free").cast("integer"))
reviews_df_processed = reviews_df_processed.withColumn("written_during_early_access", f.col("written_during_early_access").cast("integer"))

# Filter review dates from before September 12, 2003 (when Steam was launched)
reviews_df_processed = reviews_df_processed.filter(reviews_df_processed.timestamp_created >= '2003-09-12')
reviews_df_processed = reviews_df_processed.filter(reviews_df_processed.timestamp_updated >= '2003-09-12')

# Fixing Boolean columns
reviews_df_processed = reviews_df_processed.filter((reviews_df_processed["steam_purchase"] == 0) | (reviews_df_processed["steam_purchase"] == 1))
reviews_df_processed = reviews_df_processed.filter((reviews_df_processed["received_for_free"] == 0) | (reviews_df_processed["received_for_free"] == 1))
reviews_df_processed = reviews_df_processed.filter((reviews_df_processed["written_during_early_access"] == 0) |
                                                   (reviews_df_processed["written_during_early_access"] == 1))
reviews_df_processed = reviews_df_processed.filter((reviews_df_processed["voted_up"] == 0) | (reviews_df_processed["voted_up"] == 1))
reviews_df_processed = reviews_df_processed.withColumnRenamed("voted_up","positive_review")

# Cast to boolean
reviews_df_processed = reviews_df_processed.withColumn("steam_purchase", f.col("steam_purchase").cast("boolean"))
reviews_df_processed = reviews_df_processed.withColumn("received_for_free", f.col("received_for_free").cast("boolean"))
reviews_df_processed = reviews_df_processed.withColumn("written_during_early_access", f.col("written_during_early_access").cast("boolean"))
reviews_df_processed = reviews_df_processed.withColumn("positive_review", f.col("positive_review").cast("boolean"))

# Splits data into two dataframes, 1 with all the review metadata and 1 with only the recommendationid + appid + review
reviews_df_processed_metadata = reviews_df_processed.drop("review")
reviews_df_processed_reviews = reviews_df_processed.select("recommendationid", "appid", "author_steamid", "review")
```
#### 2.b.ii. Dataset #2: Steam Games
The following preprocessing was done on the `games_df`. In general, this involved droping nulls/duplicates and casting to the correct data types per the Kaggle documentaiton.
```
# Removes columns not relevant for this analysis
games_df_processed = games_df.drop("reviews", "header_image", "website", "support_url", "support_email", "full_audio_languages", "screenshots", "movies",
                                  "required_age", "metacritic_url", "supported_languages", "packages", "score_rank", "discount")
games_df_processed = games_df_processed.na.drop(subset=["appid"])
games_df_processed = games_df_processed.dropDuplicates(subset=["appid"])

# Cast to correct data types
games_df_processed = games_df_processed.withColumn("release_date", f.to_timestamp("release_date", "yyyy-MM-dd"))
games_df_processed = games_df_processed.withColumn("price", f.col("price").cast("double"))
games_df_processed = games_df_processed.withColumn("dlc_count", f.col("dlc_count").cast("integer"))
games_df_processed = games_df_processed.withColumn("windows", f.col("windows").cast("boolean"))
games_df_processed = games_df_processed.withColumn("mac", f.col("mac").cast("boolean"))
games_df_processed = games_df_processed.withColumn("linux", f.col("linux").cast("boolean"))
games_df_processed = games_df_processed.withColumn("metacritic_score", f.col("metacritic_score").cast("double"))
games_df_processed = games_df_processed.withColumn("achievements", f.col("achievements").cast("integer"))
games_df_processed = games_df_processed.withColumn("recommendations", f.col("recommendations").cast("integer"))
games_df_processed = games_df_processed.withColumn("user_score", f.col("user_score").cast("integer"))
games_df_processed = games_df_processed.withColumn("positive", f.col("positive").cast("integer"))
games_df_processed = games_df_processed.withColumn("negative", f.col("negative").cast("integer"))
games_df_processed = games_df_processed.withColumn("average_playtime_forever", f.col("average_playtime_forever").cast("integer"))
games_df_processed = games_df_processed.withColumn("average_playtime_2weeks", f.col("average_playtime_2weeks").cast("integer"))
games_df_processed = games_df_processed.withColumn("median_playtime_forever", f.col("median_playtime_forever").cast("integer"))
games_df_processed = games_df_processed.withColumn("median_playtime_2weeks", f.col("median_playtime_2weeks").cast("integer"))
games_df_processed = games_df_processed.withColumn("pct_pos_total", f.col("pct_pos_total").cast("integer"))
games_df_processed = games_df_processed.withColumn("peak_ccu", f.col("peak_ccu").cast("integer"))
games_df_processed = games_df_processed.withColumn("num_reviews_total", f.col("num_reviews_total").cast("integer"))
games_df_processed = games_df_processed.withColumn("pct_pos_recent", f.col("pct_pos_recent").cast("integer"))
games_df_processed = games_df_processed.withColumn("num_reviews_recent", f.col("num_reviews_recent").cast("integer"))
```
#### 2.b.iii. Verifying the Compatability of the Datasets
The two datasets were evaulated to ensure compatability use the following code that compares `appid` and `name` of the games.
```
# Verify that appid's match between the two datasets
joined = reviews_df_processed_metadata.select("appid", "game").alias("df1").join(
    games_df_processed.select("appid", "name").alias("df2"), on="appid", how="inner").filter(
    "df1.game != df2.name")
joined.show()
# Dropping the game name from the first dataset (100+ Million Reviews), will use the game name from the second dataset (Steam Games)
reviews_df_processed_metadata = reviews_df_processed_metadata.drop("game")
```
#### 2.b.iv. Text Processing
The review text of the top 100 most reviewed games was processed using a tokenizer and then `Word2vec`.
```
# Filtering Data to include only the Top 100 Most-Reviewed Games
review_counts = reviews_df_processed_metadata.groupBy("appid").count()
top_100_games = review_counts.orderBy(f.col("count").desc()).limit(100)
top_100_appids = [row.appid for row in top_100_games.collect()]
filtered_review_df = reviews_df_processed_reviews.filter(reviews_df_processed_reviews.appid.isin(top_100_appids))

# Tokenize Reviews
tokenizer = RegexTokenizer(inputCol='review', outputCol='tokens', pattern='\\W')
reviews_tokenized_df = tokenizer.transform(filtered_review_df)
# Remove common english "stop words"
stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
reviews_tokenized_df = stopword_remover.transform(reviews_tokenized_df)
# Drop reviews that are less than 10 tokens long
reviews_tokenized_df = reviews_tokenized_df.filter(size(reviews_tokenized_df.tokens) >= 10)

# Use Word2Vec to vectorize the reviews into vectors of size 50
word2vec = Word2Vec(vectorSize=50, minCount=1, inputCol='filtered_tokens', outputCol='review_embeddings')
word2vec_model = word2vec.fit(reviews_tokenized_df)
reviews_embeddings_df = word2vec_model.transform(reviews_tokenized_df)
```
Then average vectors for each game and author were created using the following code.
```
# Functions to average vectors together
def avg_vectors(vectors):
    if vectors:
        return np.mean(vectors, axis=0).tolist()
    else:
        return []

avg_udf = udf(avg_vectors, ArrayType(FloatType()))

# Group by game (appid) and calculate average vectorized review
game_embeddings_df = (
    reviews_embeddings_df
    .groupBy("appid")
    .agg(collect_list("review_embeddings").alias("all_review_embeddings"))
    .withColumn("game_embedding", avg_udf("all_review_embeddings"))
    .select("appid", "game_embedding")
)

# Group by review author (author_steamid) and calculate average vectorized review
author_embeddings_df = (
    reviews_embeddings_df
    .groupBy("author_steamid")
    .agg(collect_list("review_embeddings").alias("all_review_embeddings"))
    .withColumn("author_embedding", avg_udf("all_review_embeddings"))
    .select("author_steamid", "author_embedding")
)
```
### 2.c. Data Exploration
Summary statistics and exploratory plots were made using the processed data and the following code.
```
# Total # Reviews, Total # Games Reviewed, and Total # Unique Reviewers
print(f"Number of Reviews: {reviews_df_processed_metadata.count()}")
num_games_reviewed = reviews_df_processed_metadata.select("appid").distinct().count()
print(f"Number of Games Reviewed: {num_games_reviewed}")
num_unique_reviewers = reviews_df_processed_metadata.select("author_steamid").distinct().count()
print(f"Number of Unique Reviewers: {num_unique_reviewers}")
# Date range of Reviews
print("Date Range of Reviews: ")
reviews_df_processed_metadata.select(f.min("timestamp_created").alias("earliest_review"),
                                     f.max("timestamp_created").alias("latest_review")).show()
```
```
# Review Volume Over Time
# Reduce data to pandas dataframe
reviews_by_month = reviews_df_processed_metadata.withColumn("year_month", f.date_format("timestamp_created", "yyyy-MM"))
review_counts_by_month = reviews_by_month.groupBy("year_month").count().orderBy("year_month")
review_counts_pd = review_counts_by_month.toPandas()
# Line plot - Review Volume Over Time
plt.figure(figsize=(12,6))
plt.plot(review_counts_pd["year_month"], review_counts_pd["count"], marker="o")
plt.xticks(ticks=review_counts_pd.index[::6], labels=review_counts_pd["year_month"][::6], rotation=45, ha="right")
plt.xlabel("Month")
plt.ylabel("Number of Reviews")
plt.title("Monthly Review Volume Over Time")
plt.tight_layout()
plt.show()
```
```
# Plot Most-Reviewed Games (Top 10) - Bar Chart
# Reduce data to pandas dataframe
review_counts = reviews_df_processed_metadata.groupBy("appid").count()
review_counts_named = review_counts.join(games_df_processed.select("appid", "name"), on="appid", how="left")
top_10_games = review_counts_named.orderBy(f.col("count").desc()).limit(10)
top_10_games_pd = top_10_games.toPandas()
# Plot bar chart
plt.figure(figsize=(10,6))
sns.barplot(data=top_10_games_pd, x="name", y="count")
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Most Reviewed Games")
plt.xlabel("Game")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()
```
```
# Plot Most Prolific Reviewers (Top 10) - Bar Chart
# Reduce data to pandas dataframe
reviewer_counts = reviews_df_processed_metadata.groupBy("author_steamid").count()
top_10_reviewers = reviewer_counts.orderBy(f.col("count").desc()).limit(10)
top_10_reviewers_pd = top_10_reviewers.toPandas()
# Plot bar chart
plt.figure(figsize=(10,6))
sns.barplot(data=top_10_reviewers_pd, x="author_steamid", y="count")
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Most Prolific Reviewers")
plt.xlabel("Author's Steam ID")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()
```
```
# Top 10 Best & Worst Reviewed Games
# Reduce data to pandas dataframe
review_stats = reviews_df_processed_metadata.withColumn("positive_review", f.col("positive_review").cast("integer")) \
                    .groupBy("appid") \
                    .agg(
                        f.avg("positive_review").alias("positive_ratio"),
                        f.count("*").alias("total_reviews")
                    )
review_stats_filtered = review_stats.filter("total_reviews >= 10000")
review_stats_named = review_stats_filtered.join(games_df_processed.select("appid", "name"), on="appid", how="left")
review_stats_named = review_stats_named.filter((review_stats_named.name != "None") & (review_stats_named.name.isNotNull()))
top_10_best_games = review_stats_named.orderBy("positive_ratio", ascending=False).limit(10)
top_10_best_games_pd = top_10_best_games.toPandas()
top_10_worst_games = review_stats_named.orderBy("positive_ratio", ascending=True).limit(10)
top_10_worst_games_pd = top_10_worst_games.toPandas()

# Plot bar chart - Top 10 Best Games
plt.figure(figsize=(10,6))
sns.barplot(data=top_10_best_games_pd, x="name", y="positive_ratio")
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Best Reviewed Games")
plt.xlabel("Game")
plt.ylabel("Proportion of Positive Reviews")
plt.ylim(0.975, 1.0)
plt.tight_layout()
plt.show()

# Plot bar chart - Top 10 Worst Games
plt.figure(figsize=(10,6))
sns.barplot(data=top_10_worst_games_pd, x="name", y="positive_ratio")
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Worst Reviewed Games")
plt.xlabel("Game")
plt.ylabel("Proportion of Positive Reviews")
plt.tight_layout()
plt.show()
```
```
# Positivity Rate vs. Playtime at Time of Review
# Reduce data to pandas dataframe
reviews_binned = reviews_df_processed_metadata.withColumn("playtime_bin", f.floor(f.col("author_playtime_at_review") / 100) * 100)
reviews_binned = reviews_binned.withColumn("is_positive_int", f.col("positive_review").cast("integer"))
playtime_vs_positive = reviews_binned.groupBy("playtime_bin") \
        .agg(
            f.avg("is_positive_int").alias("positive_ratio"),
            f.count("*").alias("review_count")
        ) \
        .orderBy("playtime_bin")
playtime_vs_positive = playtime_vs_positive.limit(200)
playtime_vs_positive_pd = playtime_vs_positive.toPandas()
# Plot scatter plot - Positivity Rate vs. Playtime at Time of Review
plt.figure(figsize=(10,6))
plt.plot(playtime_vs_positive_pd["playtime_bin"], playtime_vs_positive_pd["positive_ratio"], marker="o")
plt.xlabel("Playtime at Review (minutes, binned)")
plt.ylabel("Proportion of Positive Reviews")
plt.title("Positivity Rate vs. Playtime at Time of Review")
plt.grid(True)
plt.tight_layout()
plt.show()
```
### 2.d. Model #1 - Simple Recommender System
The first model employs K-Means to cluster games into 10 groups using the average vectorized reviews. Then it uses the averaged vectorized reviews of the authors to assign them to a cluster (AKA gaming persona).
```
# Cluster games based on review embeddings by game
from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=10, seed=96, featuresCol='vectorized_reviews_by_game', predictionCol='game_cluster')
kmeans_model = kmeans.fit(game_features_df)
games_with_persona = kmeans_model.transform(game_features_df)
games_with_persona.show(5)
```
```
# Assign Users to Personas
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

centroids = kmeans_model.clusterCenters()

def closest_cluster(vec):
  vec = np.array(vec)
  sims = [np.dot(vec, np.array(c)) / (np.linalg.norm(vec) * np.linalg.norm(c) + 1e-9) for c in centroids]
  return int(np.argmax(sims))

closest_cluster_udf = udf(closest_cluster, IntegerType())
author_with_persona = author_features_df.withColumn("persona", closest_cluster_udf("vectorized_reviews_by_author"))
```
The clusters are then reduced using t-SNE and visualized using the following code.
```
# Visual games by cluster/persona
X = np.vstack(games_with_personas_pd['review_embedding_array'].values)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
games_with_personas_pd["x"] = X_tsne[:, 0]
games_with_personas_pd['y'] = X_tsne[:, 1]

plt.figure(figsize=(10,7))
scatter = plt.scatter(games_with_personas_pd['x'], games_with_personas_pd['y'], c=games_with_personas_pd["game_cluster"], cmap="tab10", alpha=0.6)
plt.title("t-SNE Projection of Games by Cluster/Persona")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="Cluster")
plt.show()
```
```
authors_with_personas_pd_ds = authors_with_personas_pd.sample(n=100, random_state=42)
X2 = np.vstack(authors_with_personas_pd_ds['review_embedding_array'].values)
X2_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X2)
authors_with_personas_pd_ds['x'] = X2_tsne[:, 0]
authors_with_personas_pd_ds['y'] = X2_tsne[:, 1]

plt.figure(figsize=(10,7))
scatter = plt.scatter(authors_with_personas_pd_ds['x'], authors_with_personas_pd_ds['y'], c=authors_with_personas_pd_ds["persona"], cmap="tab10", alpha=0.6)
plt.title("t-SNE Projection of Authors by Cluster/Persona")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="Cluster")
plt.show()
```
### 2.e. Model #2 - Predicting a Game's Popularity
The second model uses a Random Forest Regressor to try to predict a game's popularity based on the reviews and game metadata. For this model, popularity is the positive_ratio of the game amongst Steam users leaving reviews, where 0 is all negative reviews and 1 is all positive reviews.
```
# Cast boolean features in the "reviews metadata" data frame to integer
review_pop_pred = reviews_df_processed_metadata.withColumn("positive_review", f.col("positive_review").cast("integer"))
review_pop_pred = review_pop_pred.withColumn("steam_purchase", f.col("steam_purchase").cast("integer"))

# Group the reviews metadata by game (app id) and aggregate specified features
review_pop_pred = review_pop_pred.groupBy("appid").agg(
                        f.avg("positive_review").alias("positive_ratio"),
                        f.avg("author_playtime_at_review").alias("avg_playtime_at_review"),
                        f.avg("steam_purchase").alias("steam_purchase_ratio"),
                        f.count("*").alias("total_reviews")
                    )
# Filter out games that have less than 100 reviews
review_pop_pred = review_pop_pred.filter("total_reviews >= 100")

# Join the review metadata with specified games metadata
review_pop_pred = review_pop_pred.join(games_df_processed.select("appid", "price", "metacritic_score", "dlc_count","average_playtime_forever", "median_playtime_forever",
                                                                 "peak_ccu"), on="appid", how="left")

# Join the metadata with the vectorized reviews
ML_df = review_pop_pred.join(game_features_df.select("appid", "vectorized_reviews_by_game"), on="appid", how="left")

# Combine features into single "features" vector
numeric_cols = ['avg_playtime_at_review', 'steam_purchase_ratio', 'total_reviews', 'price', 'metacritic_score', 'dlc_count',
                'average_playtime_forever', 'median_playtime_forever', 'peak_ccu']

# Handle Nulls
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import when, udf

for c in numeric_cols:
    ML_df = ML_df.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c)))

@udf(VectorUDT())
def zero_vector_udf():
    return Vectors.dense([0.0] * 50)

ML_df = ML_df.withColumn("vectorized_reviews_by_game",
                         when(col("vectorized_reviews_by_game").isNull(), zero_vector_udf()).otherwise(col("vectorized_reviews_by_game")))
# Assemble Numeric Columns and Vector into a single features vector
assembler = VectorAssembler(
    inputCols=numeric_cols + ['vectorized_reviews_by_game'], outputCol='features'
)

# Establish Random Forest Regressor
rf = RandomForestRegressor(
    featuresCol='features',
    labelCol='positive_ratio',
    predictionCol='prediction',
    numTrees=100,
    maxDepth=10,
    seed=96
)
# Establish ML pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Split into training and test data
train_data, test_data = ML_df.randomSplit([0.8, 0.2], seed=96)

# Train model
model = pipeline.fit(train_data)

# Evaluate model
predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    labelCol='positive_ratio',
    predictionCol='prediction',
    metricName='rmse'
)

rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(labelCol='positive_ratio', predictionCol='prediction', metricName='r2').evaluate(predictions)

print("RMSE: ", rmse)
print("R^2 Score: ", r2)

```
## 3. Results
### 3.a. Data Importing
Data was successfully imported into spark data frames using the above code.
### 3.b. PreProcessing
#### 3.b.i. Dataset #1: Steam Reviews
![image](https://github.com/user-attachments/assets/35ee6e3b-24cb-4f24-b18f-2aefe1913043)
Figure 1. Example of processed review metadata (`reviews_df_processed_metadata`).
#### 3.b.ii. Dataset #2: Steam Games
![image](https://github.com/user-attachments/assets/ff4d483f-c2ba-4e05-974c-f386004e6a2d)
Figure 2. Example of processed games metadata (`games_df_processed`).
#### 3.b.iii. Verifying the Compatability of the Datasets
![image](https://github.com/user-attachments/assets/09714b9a-110d-4d5c-9a38-06f813810120)
Figure 3. Example of processed games metdata (`games_df_processed`).
#### 3.b.iv. Text Processing.
![image](https://github.com/user-attachments/assets/b1a3c132-648c-4f1a-8d44-d1d0d07d1f48)
Figure 4. Example of processed reviews (`reviews_embeddings_df`).
### 3.c. Data Exploration
![image](https://github.com/user-attachments/assets/bcdba426-8c36-4b5e-89e5-91bb521bd02a) <br>
Figure 5. Summary Statistics <br>
![Untitled](https://github.com/user-attachments/assets/9b1724b5-0774-4dbb-b592-1184475db7f4)
Figure 6. Monthly Review Volume Over Time <br>
![Untitled](https://github.com/user-attachments/assets/7a4afcb8-b883-42fb-9904-e7d08a259f65)
Figure 7. Top 10 Most Reviewed Games <br>
![Untitled](https://github.com/user-attachments/assets/2f39e0b6-4108-446d-a47f-da2e8275f062)
Figure 8. Top 10 Most Prolific Reviewers <br>
![Untitled](https://github.com/user-attachments/assets/5083f6e3-2fa3-4795-acb8-5d24edb5a8a3)
Figure 9. Top 10 Best Reviewed Games <br>
![Untitled](https://github.com/user-attachments/assets/6ecf1f42-eea7-465d-8b2f-4a1dfa2d0d85)
Figure 10. Top 10 Worst Reviewed Games <br>
![Untitled](https://github.com/user-attachments/assets/4799a8ea-8abf-43f2-ba55-d6598c5d7b20)
Figure 11. Positivity Rate vs. Playtime at Review <br>
### 3.d. Model #1 - Simple Recommender System
![Untitled](https://github.com/user-attachments/assets/c0f142cd-b3a4-422c-b6f3-dfbb6908a875)
Figure 12. Game Cluster Visualization <br>
![Untitled](https://github.com/user-attachments/assets/05044eda-30e6-45bd-945e-5a3f37a06071)
Figure 13. Author Cluster Visualization <br>
|index|game\_cluster|game\_count|games\_in\_cluster|
|---|---|---|---|
|0|0|1|\['Factorio'\]|
|1|1|1|\['The Elder Scrolls V: Skyrim'\]|
|2|2|15|\['The Witcher 3: Wild Hunt' 'Undertale' 'NieR:Automata™'
 'Halo: The Master Chief Collection' 'Persona 4 Golden' 'ELDEN RING'
 'Subnautica' 'Hollow Knight'
 'Divinity: Original Sin 2 - Definitive Edition' 'Half-Life: Alyx'
 'DARK SOULS™: REMASTERED' 'Sekiro™: Shadows Die Twice - GOTY Edition'
 "Baldur's Gate 3" 'OMORI' 'Red Dead Redemption 2'\]|
|3|3|25|\["Garry's Mod" 'Terraria' 'Euro Truck Simulator 2' 'Warframe'
 'Europa Universalis IV' 'Cities: Skylines' "No Man's Sky" 'BeamNG\.drive'
 'ARK: Survival Evolved' 'Satisfactory' 'Beat Saber' 'Destiny 2'
 'Titanfall® 2' 'Arma 3' 'Kerbal Space Program' 'Path of Exile'
 'Rocket League®' 'Stellaris' 'Tabletop Simulator' 'Blender' 'Fallout 4'
 'Slime Rancher' 'The Elder Scrolls V: Skyrim Special Edition'
 'PUBG: BATTLEGROUNDS' 'Battlefield™ 2042'\]|
|4|4|1|\['Doki Doki Literature Club\!'\]|
|5|5|5|\['Call of Duty®: Black Ops' 'Call of Duty®: Black Ops III' 'The Sims™ 4'
 'The Sims™ 3' 'Call of Duty®: Black Ops II'\]|
|6|6|20|\['Team Fortress 2' 'DayZ' 'War Thunder' 'Grand Theft Auto V Legacy'
 'RimWorld' 'FOR HONOR™' 'Stardew Valley' 'VRChat' 'Crush Crush'
 'Counter-Strike: Source' 'Counter-Strike 2' 'PAYDAY 2'
 'Getting Over It with Bennett Foddy' "Don't Starve Together"
 'NEKOPARA Vol\. 1' 'HuniePop' "Tom Clancy's Rainbow Six® Siege"
 'Dead by Daylight' 'Hearts of Iron IV'
 '5D Chess With Multiverse Time Travel'\]|
|7|7|20|\['Papers, Please' 'Rust' "Five Nights at Freddy's" 'DOOM' 'Squad'
 'Risk of Rain 2' 'Bloons TD 6' 'ULTRAKILL' 'Project Zomboid' 'Kenshi'
 'METAL GEAR RISING: REVENGEANCE' 'The Forest' 'Garfield Kart'
 'Deep Rock Galactic' 'Blade and Sorcery' 'Yakuza 0' 'Phasmophobia'
 'People Playground' 'Sea of Thieves: 2024 Edition' 'Walking Simulator'\]|
|8|8|2|\['Helltaker' 'Wallpaper Engine'\]|
|9|9|1|\['I Love You, Colonel Sanders\! A Finger Lickin’ Good Dating Simulator'\]|
### 3.e. Model #2 - Predicting a Game's Popularity
## 4. Discussion
### 4.a. Data Exploration
### 4.b. PreProcessing
#### 4.b.i. Dataset #1: Steam Reviews
On the `reviews_df` spark dataframe, `.printSchema()` is used to explore the features provided for each review and a count of all reviews is taken. There are two columns related to the Chinese gaming market that won't be useful for this analyis so they are removed using `.drop`. Additionally, it is identified that the dataset contains reviews in many different languages. For the purposes of this work, English language text analysis will be performed on the reviews; therefore, the data is filtered to include only the reviews in English and a new count is taken. Next, there are three unique identifier attributes that are vital for the analysis as well as the review itself. Rows with nulls values in any of these columns are dropped and rows with duplicate `recommendationid` values are also dropped. Upon review, many of the columns were imported as strings so the definitions in Kaggle are used to cast the mismatched columns to their appropriate datatypes. A brief analysis of the time related attributes indicated some incorrect dates so any rows containing `timestamp` values older than the launch date of Steam (September 12, 2003) are removed. Finally the processed reviews dataframe is split into two dataframes: one containing all of the metadata (`reviews_df_processed_metadata`) and one containing only the unique identifiers for the review and game plus the review (`reviews_df_processed_reviews`). Future work will include processing the reviews for Natural Language Processing (NLP).
#### 4.b.ii. Dataset #2: Steam Games
On the `games_df` spark dataframe, `.printSchema()` is used to explore the features provided for each game and a count of all games is taken. For this analysis, we will only be using game metadata that describes the type of game; therefore many columns not relevant are dropped. Additionally, any rows with null values or duplicate values in the unique identifier for the game (e.g., `appid`) are removed. Just like the review dataframe, many of the columns were imported as strings so the definitions in Kaggle are used to cast the mismatched columns to the correct datatypes.
#### 4.b.iii. Verifying the Compatability of the Datasets
While both of these datasets claim to use Steam's API to generate the data, they come from entirely different Kaggle contributors. We would like to combine the two datasets on the shared unique identifier for a game (`appid`), but a check is needed to verify that the `appid` values in each dataset are the same. Therefore the datasets are joined on `appid` and the `game` column from the reviews dataset is compared to the `name` column from the games dataset for a subset of entries. While there are some small differences in capitalization and exact wording, the subset appears to match and so we can trust that the `appid` columns in each dataset are compatible. For simplification, the `game` column is dropped from the reviews dataframe and only the `name` column from the games dataframe will be used moving forward.
#### 4.b.iv. Text Processing
The `reviews_df_processed_reviews` spark dataframe contains only the unique identifiers for each review (`recommendation_id`, `appid`, and `author_steamid`) and the review text (`review`). A vectorized representation of the review text is needed for later machine learning tasks. Therefore, the dataset is filtered down to include only the top 100 most-reviewed games which includes over 19,000,000 reviews. Next, the reviews are tokenized, common english "stop words" are removed, and only reviews longer than 10 tokens are kept. This still totals over 19,000,000 reviews. Finally, a `Word2Vec` model is applied to transform the review text into vectors of size 50. Because this process takes a fair amount of time to run, the resulting `reviews_embeddings_df` containing the vectorized version of each review is saved as a parquet file and all subsequent analysis began by loading in this parquet file. From here, the review vectors are aggregated by game (`appid`) and author (`author_steamid`).
### 4.c. Data Exploration
An analysis of the processed dataframes shows that there are 49,606,243 reviews spanning 96,042 games, written by 15,314,376 unique reviewers. Additionally, the reviews span a period of time from October 15, 2010 to November 3, 2023. A plot of "Monthly Review Volume Over Time" shows that the majority of reviews are from the last 4 years of this timeframe. A bar plot shows the "Top 10 Most Reviewed Games" of which the #1 most reviewed game, "Counter-Strike 2" has 4x more reviews than the #2 most reviewed game. A bar plot of the "Most Prolific Reviewers" shows that the #1 reviewer has reviewed almost 6,000 games. In the next exploratory plots, the `positive_review` tag (which is a boolean indicating whether or not the review was positive) is used to calculate "Proportion of Positive Reviews" for each game. Games with less than 10,000 reviews are filtered out so as not to skew that data and the "Top 10 Best Games" and "Top 10 Worst Games" are displayed in bar plots. Finally, a plot of the proportions of reviews that were positive and the author's playtime at the time of review are plotted for a subset of 500 games. Interestingly, games with the lowest proportion of positive reviews tend to have lower playtime by the review author at the time of review. This could mean that games are getting "review bombed" by people who haven't even played the game or it could also mean that people don't tend to play a game for very long before giving it a poor review.
### 4.d. Model #1 - Simple Recommender System
### 4.e. Model #2 - Predicting a Game's Popularity
The goal of the first model is to predict a game's popularity with Steam users using other provided metadata about the game. For this model, popularity is the `positive_ratio` of the game amongst Steam users leaving reviews, where 0 would be all negative reviews and 1 would be all positive reviews. <br>
First, feature engineering is performed in order to prepare the data for machine learning. This involves casting boolean features to integers, aggregating review data by game, and joining specified columns from the games metadata dataframe and the vectorized reviews. In this process, null values were identified which would cause an error when training a model; therefore, they are replaced with 0's. Finally, an `assembler` is established that will combine the numeric columns and vectorized reviews into a single `features` vector. <br>
In the final section of this code, a Random Forest Regressor model is established, trained on a subset of the data (~80%), and tested on the remaining witheld data (~20%). While the RMSE value of 0.1974 on a scale of 0-1 doesn't seem too bad, an R^2 value of 0.1136 indicates that the model is performing quite poorly and further development is needed.
## 5. Conclusion
This first model attempt indicates that it may not be possible to predict game performance given the metadata we have access to; however, there is still room for trying to improve the model by including more features, trying a different model framework (i.e., XGBoost), and tuning the model parameters. In addition to trying to improve this predictive model, in Milestone 4, we will attempt to build a simple recommender system using clustering as this dataset may be more suited for unsupervised learning.
## 6. Statement of Collaboration
Jillian O'Neel - Project Lead - Responsible for all coordination, coding, and write-ups
