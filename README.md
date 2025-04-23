# Predicting Recipe Ratings By Tag Combinations
by Vasu Banga (vbanga@umich.edu)
**Final Project for EECS 398 at the University of Michigan**

## Introduction
In our dataset, there is a multitude of information to go off of. It's relatively easy to take quantitative data, like time or steps, and predict how good a dish is going to be (For example, a dish that takes 2 minutes to make is probably not going to be as good as a dish that takes 40 minutes to make). Instead of exploring a quantitative to quantitative problem, I was more interested in seeing how different tags affect the ratings. For example, do recipes with the tag 'indian' tend to be rated higher than recipes with the 'american' tag? How do they compare?

In this prediction process, we combined our recipe data tables and our review data tables to develop a merged dataset of 234,429 entries over 83,782 recipes. With this dataframe, we are now ready to explore recipe ratings without having to look at separate datasets. With this, we now look at the following tags:

**id**: Unique to each individual recipe, allows us to distinguish which recipe a review refers to, as for each recipe, there may be multiple rows per review

**tags**: This is an array of strings, and for each recipe, contains all of the tags the author of that recipe chose to classify their recipe with.

**rating**: This is based on the reviews database. Gives a value between 1 and 5 that the user decided to rate the recipe on.

These specific columns are going to be used in our models, and all other columns won't be particularly necessary. We also will make a column named **average_rating**, which will represent the average rating for each unique recipe based on the ratings provided by users

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning

To prepare the data for modeling, we addressed several key issues in the raw dataset:

- **Replacing 0 ratings with NaN:** In the reviews data, a rating of 0 means the user submitted a comment without assigning a score. This value is automatically inserted by the system. These ratings were not meaningful, so we replaced all `rating = 0` values with `NaN`. This helped us avoid an artificial skew in our model and our ratings, and with 15,036 NaN rating values, this greatly helped what our future model could have been.

- **Removing rows with missing ratings:** After this replacement, we dropped all rows where the `rating` was `NaN` before training any models. This ensured our model only trained on complete and valid data.

- **Creating `average_rating`:** Since each recipe could have multiple reviews, we created an `average_rating` column that captures the mean rating per unique recipe. This lets us evaluate recipe-level performance when needed, and gives us flexibility in comparing model predictions to user consensus.

These steps helped us clean and align our dataset with the specific goal of predicting how combinations of tags influence recipe ratings. Our relevant dataframe now looks like this:

|     id | tags                                                                                                                                                                                                                        |   rating |   average_rating |
|-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|-----------------:|
| 333281 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] |        4 |                4 |
| 453467 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               |        5 |                5 |
| 306168 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |        5 |                5 |
| 306168 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |        5 |                5 |
| 306168 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        |        5 |                5 |

### Exploratory Analysis

#### Univariate Analysis

To understand the distribution of our key variables, we began by examining the `rating` and `average_rating` columns.

- **`rating`** represents individual user scores and is generally skewed toward higher values (4–5), suggesting that users tend to rate recipes favorably.
- **`average_rating`** represents the mean of all ratings per recipe. This still maintains a right skew as well.

 <iframe
 src="average_rating_distribution.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

 As you can see, average ratings tend to linger between 4 and 5, so we should predict most combinations of tags to produce ratings between 4 and 5.

We also explored the `tags` column, which contains arrays of descriptors like `'indian'`, `'vegan'`, `'no-cook'`, and `'weeknight'`. Some tags are very common (e.g., `'60-minutes-or-less'`), while others are niche. Tag frequency is important because high-frequency tags may dominate model training and introduce bias unless properly handled (e.g., via TF-IDF weighting or frequency capping).

Upon looking at TF-IDF values and ratings per tag, we found the top 10 best and worst tags for ratings, as shown below:

***Best Tags:***

| tag       |   weighted_avg_rating |   avg_tfidf_value |   num_recipes_using_tag |
|:----------|----------------------:|------------------:|------------------------:|
| recipe    |               4.84495 |          0.487228 |                      44 |
| catfish   |               4.83797 |          0.441213 |                      33 |
| contest   |               4.83788 |          0.47808  |                      43 |
| ragu      |               4.83788 |          0.47808  |                      43 |
| memorial  |               4.80075 |          0.341394 |                      50 |
| papaya    |               4.79168 |          0.393503 |                      51 |
| non       |               4.78863 |          0.376826 |                     302 |
| alcoholic |               4.78863 |          0.376826 |                     302 |
| trout     |               4.74528 |          0.441057 |                      25 |
| labor     |               4.72843 |          0.347367 |                      53 |


***Worst Tags***

| tag       |   weighted_badness_score |   avg_tfidf_value |   num_recipes_using_tag |
|:----------|-------------------------:|------------------:|------------------------:|
| year      |                 0.879404 |          0.4347   |                      11 |
| libyan    |                 0.769999 |          0.452825 |                      15 |
| beijing   |                 0.736217 |          0.545127 |                      10 |
| tempeh    |                 0.702352 |          0.483691 |                      15 |
| icelandic |                 0.693965 |          0.488462 |                      25 |
| sugar     |                 0.690432 |          0.500882 |                      19 |
| georgian  |                 0.670643 |          0.446912 |                      11 |
| honduran  |                 0.624092 |          0.458962 |                      11 |
| costa     |                 0.618077 |          0.414555 |                      18 |
| angolan   |                 0.603745 |          0.567864 |                       9 |

#### Bivariate Analysis

To explore how tags influence ratings, we conducted a bivariate analysis between `tags` and `average_rating`.

We grouped recipes by tag and computed the distribution of average ratings per tag. Below is a box plot showing the spread of `average_rating` across the **top 10 most common tags**.

This analysis helped identify which tags are generally associated with higher or lower-rated recipes.

 <iframe
 src="average_rating_for_top_10_tags.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

#### Notes
***Imputation***: There was really no way for me to impute the value of missing ratings, and if I were to base it off the comment, this would result in an entire LLM being run on every comment to predict where it should end up, or something of the sort. Ultimately, this did not seem particularly necessary, and with an abundance of data to go off of, there was no need for Imputation

***Interesting Aggregates***: Some of the least rated and most rated tags seemed particularly interesting to me, including `recipe`, `ragu` both being highly rated with high TF-IDF values, while `sugar` was rated particularly poorly 

***Tag Handling***: If you look carefully at certain tags, some tags have been split into seperate tags, although they should clearly be paired together. Due to the nature of these splits, and trying to differentiate between an intentional split and not a split, this did not seem particularly important to search through. This is the reason for `non` and `alcoholic` being separate in the good tags

## Framing a Prediction Problem

For the prediction problem, I chose to predict the rating of a recipe based on the tags, which results out to a regression problem. This is due to `rating` being able to be any integer between 1 and 5, and with most values resulting between 4 and 5, decimal places are particularly important here, so we cannot simply generalize.

To evaluate the model's effectiveness, we look towards the Mean Squared Error value (MSE). There are a few reasons for this

- **Interpretability**: Tells us exactly how off our predictions are. With our model predicting only between values 1 - 5, we need to make sure this value is low, and MSE allows us to check that

- **Sensitive**: MSE penalizes large errors more heavily, which allows our model to prioritize minimizing large mistakes. With a training set that's so skewed towards 4 and 5, this makes our final model more sensitive, which is perfect for us.

- **Stability**: All of our outputs are between 1-5, so the common issue with using mean as a metric - outliers - is not an issue here. We know that the furthest an outlier can go is 1 or 5, so these cases are not disproportionately penalized.

When developing our model, I chose to use the following metrics: `rating` and `tags`. When finally evaluating the model's effectiveness, I compared to `average_rating` and `rating`, to get a more accurate understanding of how accurate the model truly is, and how much outliers truly affected this model.

## Baseline Model

For our baseline model, we looked towards two different models - **Linear Regression** and **Random Forest**

*Note*: We did consider additional models, however other issues were faced, including complexity, extreme overfitting, extreme underfitting, and more. Linear Regression and Random Forest resulted in the best baseline models, so we chose these to begin with

- **Linear Regression**: We started with Linear Regression due to its simplicity. The baseline model only had to take one particular tag, and then use it to generate the predicted rating from just that individual tag, with no regard for TF-IDF value in the recipe it's being trained on. After encoding inputs with **One-hot encoding**, we were able to convert the categorical inputs into quantitative data for our model to evaluate on. This proved itself to be quite accurate, with a testing MSE of 0.235

- **Random Forest**: With RandomForest, we dived right in to our final model, which takes multiple tags and predicts the combined rating. In order to accomplish this, we utilized a **MultiLabelBinarizer**, which takes all the tags, and encodes it to 0s and 1s in an array to be used as an input into this model. This allowed us to take something categorical and convert it into numerical data for this model to converge on. This particular model resulted in a MSE of 0.5449, which is incredibly large, much worse than linear regression. However, when compared to the average rating, we ended up with a MSE of 0.08, meaning the model is incredibly robust to outliers, meaning we generated an incredibly robust model.

Ultimately, both models had somewhat good performance, however the intentions were different for both. Linear Regression only checked a single tag, and then used that to predict a rating, while Random Forest used multiple tags to generate a combined value, and wasn't particularly good at handling individual tags, since that wasn't what it was trained on.

When moving to the final model, we had to convert our LinearRegression model to something that could handle multiple tags.

## Final Model

To begin our Final Model, we had to make our Linear Regression model able to handle multiple inputs, and here's how we approached it:

- **Averaging**: At first, we began with taking each tag, averaging out the outputs, and then returning the mean of the ratings. While this worked somewhat well, this valued meaningless tags too highly, which prompted us to stray away from this

- **TF-IDF combination**: This model took the input, ran a TF-IDF to get a value for each input, and multiplied this value by the predicted output from our barebones model, which resulted in an incredibly accurate model, with an MSE of 0.2465 when compared to average ratings.

### Comparison

Ultimately, when comparing the two models, I elected to go towards RandomForest. With an MSE against average ratings of 0.08, this was incredibly accurate. This is most likely due to the very nature of binary trees, which can detect relationships between various tags, and value those, along with their standalone values. For example, something with the tags `southern` and `baked` may be valued differently than something with the tags `lebanese` and `baked`, and the predicted ratings from RandomForest tended to reflect these relationships much more than our TF-IDF combinations did.

Regarding performance, our **Test MSE** was 0.5449, while our MSE on **Average Ratings** was 0.08, meaning our model is incredibly robust in its predictions, and particularly accurate as well. I did run my model on some edge cases, including every tag as well, and it was able to hold up particularly well, penalizing random assortments of tags, and rated well chosen tags much more highly.