# Comparing Nutrition In European vs. American Recipes
Analysis on recipes from food.com

By Nayan Bhatnagar

## Introduction
There are many different types of dishes from different cultures around the world, with these dishes often having different characteristics, such as their nutritional value. But this brings up an important question, are certain types of dishes healthier than others? This is especially interesting considering the different health outcomes in different countries. For example, the [World Health Organization](https://www.who.int/data/gho/data/indicators/indicator-details/GHO/prevalence-of-obesity-among-adults-bmi-=-30-(age-standardized-estimate)-(-)) estimated the adult obesity rate in the US to be 42% in 2022, while estimating it to be less than 25% in most European countries. Using recipe data from [food.com](https://www.food.com), I asked whether recipes tagged as American would be unhealthier than those tagged as European, which could help people make healthier choices when trying new recipes and choosing what types of dishes to eat. The data comes from two datasets, one with recipes (with 83,782 rows), and another with the reviews for the recipes (with 731,927 rows). The most important columns are as follows:
- name: The name of the recipe
- minutes: The estimated minutes a recipe takes
- tags: The tags assigned to the recipe
- nutrition: Nutrition information in terms of calories and daily value percent for different nutrients
- n steps: The number of steps a recipe takes
- rating: The rating (1 to 5) given in a review of the recipe

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning and Merging
To process the data, both datasets were cleaned and then merged in the following ways:
1. Unnecessary columns were dropped (contributer id and user id)
2. The recipe and review dates were converted into datetime
- This allowed for analysis to be done based on time, since these columns were strings before
3. Minute values of 0 and ratings of 0 were replaced with NaN
- The only recipe with 0 minutes did not have a time listed on food.com
- Comments with rating values of 0 represent comments on food.com without the user selecting a rating, so the rating would be a missing value
4. Separate columns from the nutrition column were created
- The nutrition column is a string of nutrition values, so this string is split into different columns to extract the nutrition data
5. Columns for the composition of each recipe by each nutrition value were created
- One problem with the nutrition data is it increases based on the serving size of the recipe, which can be an arbitary value (since the serving size of a dish can be set to any amount)
- To be able to compare recipes with varying serving sizes, the nutrition values are first converted into calories from their daily value percent (except sodium, which does not have any calories, so is not included in this calculation)
- Then, that value is divided by the calories in the recipe, to get the proportion of a recipe made up of a certain nutrient (for example, if the sugar composition is 0.5, it means half of the calories in that recipe come from sugar)
- Note that this means that recipes with 0 calories will have their nutrition composition values set to NaN, since the calorie composition proportion can't be found if there are no calories in the recipe
6. Two additional columns were created, one for if a recipe has the tag American, and another if it has the tag European
- This is necessary to compare American and European dishes
7. The recipes data was left merged with the reviews data to get a combined dataset
8. This was used to add an average rating column, which has the mean rating score of each recipe
- This allows comparing recipes by their average rating
<br>
Here are all the columns for the merged data:<br>
<br>

| Column         | Type           |
|:---------------|:---------------|
| name           | object         |
| id             | int64          |
| minutes        | float64        |
| recipe date    | datetime64[ns] |
| tags           | object         |
| n steps        | int64          |
| steps          | object         |
| description    | object         |
| ingredients    | object         |
| n ingredients  | int64          |
| calories       | float64        |
| fat            | float64        |
| sugar          | float64        |
| sodium         | float64        |
| protein        | float64        |
| sat fat        | float64        |
| carbs          | float64        |
| fat %          | float64        |
| sugar %        | float64        |
| protein %      | float64        |
| sat fat %      | float64        |
| carbs %        | float64        |
| usa            | bool           |
| euro           | bool           |
| review date    | datetime64[ns] |
| rating         | float64        |
| review         | object         |
| average rating | float64        |

<br>
Here is the head of the dataframe, but only showing the more important columns that will be used:<br>
(scroll right to see all)
<br><br>

| name                                 |   minutes |   n steps | ingredients                                                                                                                                                                    |   calories |   sugar prop |   sat fat prop | usa   | euro   |   rating |
|:-------------------------------------|----------:|----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|-------------:|---------------:|:------|:-------|---------:|
| 1 brownies in the world    best ever |        40 |        10 | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa... |      138.4 |     0.362717 |       0.248345 | False | False  |        4 |
| 1 in canada chocolate chip cookies   |        45 |        12 | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour...                    |      595.1 |     0.355981 |       0.155031 | False | False  |        5 |
| 412 broccoli casserole               |        40 |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground...          |      194.8 |     0.030924 |       0.334312 | False | False  |        5 |
| 412 broccoli casserole               |        40 |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground...          |      194.8 |     0.030924 |       0.334312 | False | False  |        5 |
| 412 broccoli casserole               |        40 |         6 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground...          |      194.8 |     0.030924 |       0.334312 | False | False  |        5 |

### Univariate Analysis
This plot shows the distribution of recipes by the proportion of how much of the recipe's calories come from sugar. It is clearly a unimodal and right skewed distribution, meaning that most recipes only have a sugar composition of around 10% or less, and that the counts decrease as the sugar proportion increases.

<iframe
  src="assets/sugar-count.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Bivariate Analysis
Here, there are two boxplots comparing the distribution of sugar composition between recipes tagged as European, and those tagged as American. The distributions look similar, though the American sugar mean is around 0.05 higher than the European, and has a much higher third quartile, meaning there are more American recipes with extremely high sugar values.

<iframe
  src="assets/sugar-usa-vs-euro.html"
  width="600"
  height="800"
  frameborder="0"
></iframe>

### Interesting Aggregates
The table allows us to see the mean sugar composition proportion based on whether a recipe is American, European, neither, or both. Interestingly, recipes which are neither have the highest proportion, but recipes that are only American have a similar proportion as well. Also, European recipes have the lowest sugar proportion, though this increases if the recipe is also American.

| usa   |    False |     True |
|:------|---------:|---------:|
| False | 0.18082  | 0.117803 |
| True  | 0.169283 | 0.149865 |

## Assessment of Missingness

### MNAR Analysis
I believe the rating column of the data might be missing not at random, meaning based on the true values itself. This is because it is reasonable to think that people who do not give a rating might feel differently about a recipe than those who do, such as by not feeling strongly about the recipe (meaning giving a rating that is in the middle). Most likely, people who strongly reject or approve of a recipe will rate it more often than those who don't. However, it's also possible that people don't give a review until they try the recipe for themselves, so data on whether a commenter tried to make the recipe first could help explain the missingness and make it missing at random instead.

### Missingness Dependency
Two permutation tests (at a significance level of 0.01) were done, which tested if the missingness of rating was associated with the mean number of steps in the recipes, and with the mean number of minutes recipes took. Both tests used absolute difference in means to compare recipes without a rating and those with a rating. The test with the number of steps returned a p-value of 0, so it's highly likely that the missingness of rating is dependent on the number of steps. However, the test with minutes returned a p-value of 0.1192, which is higher than 0.01, so the missingness of rating likely does not depend on the minutes a recipe takes.

These plot shows the empirical distribution of the test statistics from the permutation test, and the observed statistics as well.

<iframe
  src="assets/missingness-n-steps.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<br>

<iframe
  src="assets/missingness-minutes.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Hypothesis Testing
Since I am interested in comparing the nutrition values of American and European dishes, I performed a permutation test comparing the mean sugar composition of American and European recipes.

Null Hypothesis: The average percent of calories coming from sugar in recipes tagged as *American* is **equal** to those tagged as *European*.

Alternate Hypothesis: The average percent of calories coming from sugar in recipes tagged as *American* is **higher** than those tagged as *European*.

The test is a permutation test since I am comparing the distributions of two samples, American and European recipes. I am using a significance level of 0.01 to be more certain before deciding to reject the null hypothesis, and using the difference in means as the test statistic, since my alternate hypothesis checks if the mean for American recipes is higher than for European ones, since my overall question is if American dishes are unhealthier.

The resulting p-value is 0, so the null hypothesis can be rejected. This means that it is likely that American recipes do have a higher mean sugar composition than European recipes. The plot shows the empirical distribution of the test statistics which were generated, and the observed statistic.

<iframe
  src="assets/test-sugar-diff.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Framing a Prediction Problem
While so far I have been comparing American and European dishes, now the recipes data can be used to predict if a recipe is American or European, which can help show how much the features in the data help to differentiate between the two, and also can help food.com automatically tag recipes after they have been posted if the user who submitted them doesn't. I will be predicting if a recipe is American or European (excluding recipes which are neither or both) using multiclass classification, since neither type will be considered positive or negative. Accuracy will be used to evaluate the model, since both groups have around the same frequency, so are balanced, and because using F1-score would require labeling either American or European as positive, but I don't want to give extra importance to either group.

The final model ends up using a recipe's ingredients and its nutrition information, which can be reasonably be known before making a prediction, since the recipe itself, and its nutrition facts can be seen before deciding what tags it should be labeled with.

## Baseline Model
The baseline model is a Random Forest Classifier that uses the calories, sugar composition, and saturated fat composition of a recipe, which are all quantitative features. I used these features since I predict that the main difference between American and European foods is in their healthiness, and these features can help estimate that, especially since the prior hypothesis test suggested a difference in sugar composition between American and European recipes.

The features were used as is, and the classifier used entropy as its evaluation criterion, and a max depth of 5. The model returned a training accuracy of around 58%, and a testing accuracy of around 53.4%, which are both very poor results, since the baseline accuracy would be around 50% (if every recipe is guessed as the same class). Therefore, the model is currently bad as it did not do much better than the baseline on the testing data.

## Final Model
The final model also uses a Random Forest Classifier with calories, sugar composition, and saturated fat composition as features, but also with the additional features of ingredients and protein composition. I added protein composition since it can be used as an additional way of measuring healthiness, especially if one class of recipes tend to have more recipes with meat or protein-rich plants. I also added ingredients since it's reasonable to assume Americans most likely have different ingredients available to them than Europeans, leading to recipes with different ingredients. Protein composition is quantitative, so it was left as is, but since the list of ingredients is nominal, I used a bag of words model on it using Count Vectorizer, in which each ingredient represented a token.

Then, I used Grid Search CV to search for optimal hyperparameters. First, Grid Search CV searched for the best evaluation criterion between Gini impurity and entropy, since I wanted to see which would measure the quality of splits better. Next, it searched for the best max depth for the trees out of these values: [2, 5, 10, 20, 40, 60, 80, 100, 120]. I tested smaller depth values to first make sure the model wouldn't overfit, and then tested much higher ones to make sure the model wouldn't underfit.

The final model gave entropy and a max depth of 100 as the best hyperparameters, and returned an accuracy of 99.1% on the training data, and 77.6% on the testing data. Therefore, this model did much better due to its new test accuracy 77.6% being much higher than the original test accuracy of 53.4%. Also, the line plot below shows how the model balanced between underfitting and overfitting based on max depth to find the optimal value (the accuracy it shows is only using entropy).

<iframe
  src="assets/acc-by-max-depth.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

## Fairness Analysis
I decided to test fairness based on the minutes a recipe takes, by splitting recipes into quicker ones (minutes is at least the median minutes), and into slower ones (minutes is more than the median minutes).

Null Hypothesis: The model is **fair** for *quicker* and *slower* recipes, meaning that its accuracy is **similar** for both groups, and differences are due to random chance.

Alternate Hypothesis: The model is **unfair**, meaning that its accuracy is **lower** for *quicker* recipes which have lower minute values than for *slower* recipes which have higher minute values.

I used the difference in accuracy means (slow recipes accuracy - quick recipes accuracy) as the test statistic, since I suspected that quicker recipes would be simpler, causing them to have less ingredients and nutrition information, making the model potentially perform worse on them. I also used a significance level of 0.01 to make sure to only reject the null when there is lots of evidence that it might be false, and kept accuracy as the evaluation metric, since the proportion of recipes of each type remain close to 0.5 for both quick and slow recipes.

The test returned a p-value of 0.03, which is more than the significance level of 0.01, so the null hypothesis can't be rejected. This means there isn't enough evidence to suggest that the model performs worse for quicker recipes than for slower ones. The plot below shows the distribution of the test differences in accuracy, and the observed difference as well.

<iframe
  src="assets/minutes-fairness-test.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>





