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


## Framing a Prediction Problem

## Baseline Model

## Final Model
