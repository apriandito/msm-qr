
# Load Packages and Set Parameter -----------------------------------------

library(tidyverse)
library(skimr)
library(tidymodels)
library(randomForest)
library(discrim)
library(vip)

# Set Parameter
tidymodels_prefer()
set.seed(1234)

# Load Data ---------------------------------------------------------------

# Load Data
df <- read_csv("data/titanic.csv")

# Explore Data ------------------------------------------------------------

# Show Data
df

# Show Variable
glimpse(df)

# Show Data Information
skim(df)

# Exploration using Visualization (Histogram)
df %>%
  ggplot(aes(x = Age, fill = as.factor(Survived))) +
  geom_histogram(color = "white")

# Exploration using Visualization (Bar Chart)
df %>%
  group_by(Sex, Survived) %>%
  count() %>%
  ggplot(aes(x = Sex, y = n, fill = as.factor(Survived))) +
  geom_col(position = "dodge")


# Split Data --------------------------------------------------------------

# Split Data to Training and Testing
df_split <- initial_split(df, prop = 0.7)
df_split

# Create Fold Validation
folds <- vfold_cv(training(df_split), v = 5, strata = Survived)


# Set Target and Feature --------------------------------------------------

# Create Recipe
df_recipe <- df %>%
  recipe(Survived ~ Sex + Pclass + Age + SibSp + Parch + Fare) %>%
  step_mutate_at(Survived, fn = factor, skip = TRUE) %>%
  step_impute_median(all_numeric())

# Set Model ---------------------------------------------------------------

# Define Decision Tree Model
dt <- decision_tree(tree_depth = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# Define Random Forest Model
rf <- rand_forest(trees = tune()) %>%
  set_engine("randomForest") %>%
  set_mode(mode = "classification")

# Define ANN Model
nn <- mlp(hidden_units = tune()) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# Define Naive Bayes Model
nb <- naive_Bayes(smoothness = tune(), Laplace = tune()) %>%
  set_engine("naivebayes")


# Set Workflow ------------------------------------------------------------

# Put in the workflow
all_workflows <- workflow_set(
  preproc = list(df_recipe),
  models = list(dt, rf, nn, nb)
) %>%
  workflow_map(fn = "tune_grid", resamples = folds, verbose = TRUE)

# Rank Result
rank_results(all_workflows, rank_metric = "roc_auc")

# Plot Rank Result
autoplot(all_workflows, metric = "roc_auc")


# Finalize ----------------------------------------------------------------

# Select Best Parameter
best_parameter <- all_workflows %>%
  extract_workflow_set_result(id = "recipe_rand_forest") %>%
  select_best(metric = "roc_auc")

# Get Final Workflow
final_workflow <- all_workflows %>%
  extract_workflow("recipe_rand_forest") %>%
  finalize_workflow(best_parameter) %>%
  fit(training(df_split))

# Variable Importance Plot
final_workflow %>%
  extract_fit_parsnip() %>%
  vip()


# Evaluation --------------------------------------------------------------

# Make Prediciton to Test Data
testing(df_split) %>%
  bind_cols(final_workflow %>% predict(testing(df_split))) %>%
  relocate(Survived, .pred_class)

# Confusion Matrix
testing(df_split) %>%
  mutate(Survived = as_factor(Survived)) %>%
  bind_cols(final_workflow %>% predict(testing(df_split))) %>%
  conf_mat(truth = Survived, estimate = .pred_class)

# Define metric for evaluation
multi_metrics <- metric_set(
  accuracy,
  sensitivity,
  specificity,
  recall,
  f_meas
)

# See Performance Evaluation
testing(df_split) %>%
  mutate(Survived = as_factor(Survived)) %>%
  bind_cols(final_workflow %>% predict(testing(df_split))) %>%
  multi_metrics(truth = Survived, estimate = .pred_class)

# Make Prediction ---------------------------------------------------------

# Load Data
df_new <- read_csv("data/titanic-prediction.csv")

# Prediction Result
df_new %>%
  bind_cols(final_workflow %>% predict(df_new))
