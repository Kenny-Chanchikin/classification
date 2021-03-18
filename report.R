install.packages("tidyverse")
install.packages("data.table")
install.packages("mlr3verse")
library("tidyverse")
library("ggplot2")
library("skimr")
library("DataExplorer")
library("data.table")
library("mlr3verse")
library("xgboost")

set.seed(212) # set seed for reproducibility

#Load data
hotels <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv")

#Initial data
hotels <- hotels %>%
  filter(adr < 4000) %>% 
  mutate(total_nights = stays_in_weekend_nights+stays_in_week_nights)

hotels <- hotels %>%
  select(-reservation_status, -reservation_status_date) %>% 
  mutate(kids = case_when(
    children + babies > 0 ~ "kids",
    TRUE ~ "none"
  ))

hotels <- hotels %>% 
  select(-babies, -children)

hotels <- hotels %>% 
  mutate(parking = case_when(
    required_car_parking_spaces > 0 ~ "parking",
    TRUE ~ "none"
  )) %>% 
  select(-required_car_parking_spaces)

hotels.bycountry <- hotels %>% 
  group_by(country) %>% 
  summarise(total = n(),
            cancellations = sum(is_canceled),
            pct.cancelled = cancellations/total*100)

hotels <- hotels %>% 
  mutate(is_canceled = case_when(
    is_canceled > 0 ~ "YES",
    TRUE ~ "NO"
  ))

hotels.par <- hotels %>%
  select(hotel, is_canceled, kids, meal, customer_type) %>%
  group_by(hotel, is_canceled, kids, meal, customer_type) %>%
  summarize(value = n())

hotels2 <- hotels %>% 
  select(-country, -reserved_room_type, -assigned_room_type, -agent, -company,
         -stays_in_weekend_nights, -stays_in_week_nights)

# Define factor
hotels2$hotel <- as.factor(hotels2$hotel)
hotels2$arrival_date_month <- as.factor(hotels2$arrival_date_month)
hotels2$meal <- as.factor(hotels2$meal)
hotels2$market_segment <- as.factor(hotels2$market_segment)
hotels2$distribution_channel <- as.factor(hotels2$distribution_channel)
hotels2$deposit_type <- as.factor(hotels2$deposit_type)
hotels2$customer_type <- as.factor(hotels2$customer_type)
hotels2$kids <- as.factor(hotels2$kids)
hotels2$parking <- as.factor(hotels2$parking)
hotels2$is_canceled <- as.factor(hotels2$is_canceled)

# Visualize Data
skimr::skim(hotels2)

DataExplorer::plot_bar(hotels2, ncol = 2)
DataExplorer::plot_histogram(hotels2, ncol = 2)
DataExplorer::plot_boxplot(hotels2, by = "is_canceled", ncol = 2)


#Define task
hotel_task <- TaskClassif$new(id = "HotelCancel",
                              backend = hotels2, # <- NB: no na.omit() this time
                              target = "is_canceled",
                              positive = "YES")

# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(hotel_task)

####################################################
# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

# Create a pipeline which encodes and then fits an XGBoost model
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Now try with a model that needs no missingness
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)

# Now fit as normal ... we can just add it to our benchmark set
res <- benchmark(data.table(
  task       = list(hotel_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    pl_xgb,
                    pl_log_reg),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

learner_lr <- lrn("classif.log_reg")

library("paradox")
library("mlr3tuning")

search_space = ps(
  maxit = p_int(lower = 10, upper = 50)
)
measure = msr("classif.ce")
evals20 = trm("evals", n_evals = 20)
instance = TuningInstanceSingleCrit$new(
  task = hotel_task,
  learner = learner_lr,
  resampling = cv5,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

tuner = tnr("random_search")
tuner$optimize(instance)
instance$result_learner_param_vals
learner_lr$param_set$values = instance$result_learner_param_vals
train <- learner_lr$train(hotel_task)
pred <- train$predict(hotel_task)
result <- pred$confusion
result/rowSums(result)*100
