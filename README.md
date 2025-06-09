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
### 2.c. Data Exploration
### 2.d. Model #1 - Simple Recommender System
### 2.e. Model #2 - Predicting a Game's Popularity
The goal of the first model is to predict a game's popularity with Steam users using other provided metadata about the game. For this model, popularity is the `positive_ratio` of the game amongst Steam users leaving reviews, where 0 would be all negative reviews and 1 would be all positive reviews. <br>
First, feature engineering is performed in order to prepare the data for machine learning. This involves casting boolean features to integers, aggregating review data by game, and joining specified columns from the games metadata dataframe and the vectorized reviews. In this process, null values were identified which would cause an error when training a model; therefore, they are replaced with 0's. Finally, an `assembler` is established that will combine the numeric columns and vectorized reviews into a single `features` vector. <br>
In the final section of this code, a Random Forest Regressor model is established, trained on a subset of the data (~80%), and tested on the remaining witheld data (~20%). While the RMSE value of 0.1974 on a scale of 0-1 doesn't seem too bad, an R^2 value of 0.1136 indicates that the model is performing quite poorly and further development is needed.
## 3. Results
### 3.a. Data Exploration
### 3.b. PreProcessing
#### 3.b.i. Dataset #1: Steam Reviews
On the `reviews_df` spark dataframe, `.printSchema()` is used to explore the features provided for each review and a count of all reviews is taken. There are two columns related to the Chinese gaming market that won't be useful for this analyis so they are removed using `.drop`. Additionally, it is identified that the dataset contains reviews in many different languages. For the purposes of this work, English language text analysis will be performed on the reviews; therefore, the data is filtered to include only the reviews in English and a new count is taken. Next, there are three unique identifier attributes that are vital for the analysis as well as the review itself. Rows with nulls values in any of these columns are dropped and rows with duplicate `recommendationid` values are also dropped. Upon review, many of the columns were imported as strings so the definitions in Kaggle are used to cast the mismatched columns to their appropriate datatypes. A brief analysis of the time related attributes indicated some incorrect dates so any rows containing `timestamp` values older than the launch date of Steam (September 12, 2003) are removed. Finally the processed reviews dataframe is split into two dataframes: one containing all of the metadata (`reviews_df_processed_metadata`) and one containing only the unique identifiers for the review and game plus the review (`reviews_df_processed_reviews`). Future work will include processing the reviews for Natural Language Processing (NLP).
#### 3.b.ii. Dataset #2: Steam Games
On the `games_df` spark dataframe, `.printSchema()` is used to explore the features provided for each game and a count of all games is taken. For this analysis, we will only be using game metadata that describes the type of game; therefore many columns not relevant are dropped. Additionally, any rows with null values or duplicate values in the unique identifier for the game (e.g., `appid`) are removed. Just like the review dataframe, many of the columns were imported as strings so the definitions in Kaggle are used to cast the mismatched columns to the correct datatypes.
#### 3.b.iii. Verifying the Compatability of the Datasets
While both of these datasets claim to use Steam's API to generate the data, they come from entirely different Kaggle contributors. We would like to combine the two datasets on the shared unique identifier for a game (`appid`), but a check is needed to verify that the `appid` values in each dataset are the same. Therefore the datasets are joined on `appid` and the `game` column from the reviews dataset is compared to the `name` column from the games dataset for a subset of entries. While there are some small differences in capitalization and exact wording, the subset appears to match and so we can trust that the `appid` columns in each dataset are compatible. For simplification, the `game` column is dropped from the reviews dataframe and only the `name` column from the games dataframe will be used moving forward.
#### 3.b.iv. Text Processing.
The `reviews_df_processed_reviews` spark dataframe contains only the unique identifiers for each review (`recommendation_id`, `appid`, and `author_steamid`) and the review text (`review`). A vectorized representation of the review text is needed for later machine learning tasks. Therefore, the dataset is filtered down to include only the top 100 most-reviewed games which includes over 19,000,000 reviews. Next, the reviews are tokenized, common english "stop words" are removed, and only reviews longer than 10 tokens are kept. This still totals over 19,000,000 reviews. Finally, a `Word2Vec` model is applied to transform the review text into vectors of size 50. Because this process takes a fair amount of time to run, the resulting `reviews_embeddings_df` containing the vectorized version of each review is saved as a parquet file and all subsequent analysis began by loading in this parquet file. From here, the review vectors are aggregated by game (`appid`) and author (`author_steamid`).
### 3.c. Data Exploration
An analysis of the processed dataframes shows that there are 49,606,243 reviews spanning 96,042 games, written by 15,314,376 unique reviewers. Additionally, the reviews span a period of time from October 15, 2010 to November 3, 2023. A plot of "Monthly Review Volume Over Time" shows that the majority of reviews are from the last 4 years of this timeframe. A bar plot shows the "Top 10 Most Reviewed Games" of which the #1 most reviewed game, "Counter-Strike 2" has 4x more reviews than the #2 most reviewed game. A bar plot of the "Most Prolific Reviewers" shows that the #1 reviewer has reviewed almost 6,000 games. In the next exploratory plots, the `positive_review` tag (which is a boolean indicating whether or not the review was positive) is used to calculate "Proportion of Positive Reviews" for each game. Games with less than 10,000 reviews are filtered out so as not to skew that data and the "Top 10 Best Games" and "Top 10 Worst Games" are displayed in bar plots. Finally, a plot of the proportions of reviews that were positive and the author's playtime at the time of review are plotted for a subset of 500 games. Interestingly, games with the lowest proportion of positive reviews tend to have lower playtime by the review author at the time of review. This could mean that games are getting "review bombed" by people who haven't even played the game or it could also mean that people don't tend to play a game for very long before giving it a poor review.
### 3.d. Model #1 - Simple Recommender System
### 3.e. Model #2 - Predicting a Game's Popularity
## 4. Discussion
### 4.a. Data Exploration
### 4.b. PreProcessing
#### 4.b.i. Dataset #1: Steam Reviews
#### 4.b.ii. Dataset #2: Steam Games
#### 4.b.iii. Verifying the Compatability of the Datasets
#### 4.b.iv. Text Processing.
### 4.c. Data Exploration
### 4.d. Model #1 - Simple Recommender System
### 4.e. Model #2 - Predicting a Game's Popularity
## 5. Conclusion
This first model attempt indicates that it may not be possible to predict game performance given the metadata we have access to; however, there is still room for trying to improve the model by including more features, trying a different model framework (i.e., XGBoost), and tuning the model parameters. In addition to trying to improve this predictive model, in Milestone 4, we will attempt to build a simple recommender system using clustering as this dataset may be more suited for unsupervised learning.
## 6. Statement of Collaboration
Jillian O'Neel - Project Lead - Responsible for all coordination, coding, and write-ups
