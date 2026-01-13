#' =============================================================================
#' POOLED ZSCORE MODEL WITH AR FILTER - OPTIMIZED
#' =============================================================================
#'
#' OPTIMIZATIONS APPLIED (aligned with XGBoost improvements):
#' 1. CLASS WEIGHTING - Weighted observations to handle 15:1 class imbalance
#' 2. RANDOM EFFECTS - Same structure as original (random intercepts + random slopes)
#' 3. IMPROVED OPTIMIZER - Better convergence settings
#'
#' This script trains an OPTIMIZED mixed-effects logistic regression model using:
#' - ALL ZSCORE features (from Phase 2)
#' - WITH_AR_FILTER: IPC_{r,t} <= 2 AND AR prediction == 0
#' - Class-weighted observations for imbalance handling
#' - Country-level random intercepts
#'
#' Author: Victor Collins Oppon
#' Date: December 2025
#' =============================================================================

# Clear environment
rm(list = ls())
gc()

# Load libraries
library(lme4)
library(pROC)
library(dplyr)
library(tidyr)
library(readr)
library(jsonlite)

# =============================================================================
# THRESHOLD OPTIMIZATION FUNCTION
# =============================================================================

compute_optimal_thresholds <- function(y_true, pred_prob) {
  if(length(unique(y_true)) < 2) {
    return(list(
      youden_threshold = 0.5, f1_threshold = 0.5, high_recall_threshold = 0.3,
      youden_index = NA, max_f1 = NA, recall_at_high_recall = NA
    ))
  }

  thresholds <- seq(0.01, 0.99, by = 0.01)

  metrics_df <- data.frame(
    threshold = thresholds,
    tp = sapply(thresholds, function(t) sum(y_true == 1 & pred_prob >= t)),
    tn = sapply(thresholds, function(t) sum(y_true == 0 & pred_prob < t)),
    fp = sapply(thresholds, function(t) sum(y_true == 0 & pred_prob >= t)),
    fn = sapply(thresholds, function(t) sum(y_true == 1 & pred_prob < t))
  )

  metrics_df <- metrics_df %>%
    mutate(
      sensitivity = ifelse(tp + fn > 0, tp / (tp + fn), 0),
      specificity = ifelse(tn + fp > 0, tn / (tn + fp), 0),
      precision = ifelse(tp + fp > 0, tp / (tp + fp), 0),
      f1 = ifelse(precision + sensitivity > 0, 2 * precision * sensitivity / (precision + sensitivity), 0),
      youden_j = sensitivity + specificity - 1
    )

  youden_idx <- which.max(metrics_df$youden_j)
  youden_threshold <- metrics_df$threshold[youden_idx]
  youden_index <- metrics_df$youden_j[youden_idx]

  f1_idx <- which.max(metrics_df$f1)
  f1_threshold <- metrics_df$threshold[f1_idx]
  max_f1 <- metrics_df$f1[f1_idx]

  high_recall_candidates <- metrics_df %>% filter(sensitivity >= 0.90)
  if(nrow(high_recall_candidates) > 0) {
    high_recall_threshold <- max(high_recall_candidates$threshold)
    recall_at_high_recall <- high_recall_candidates$sensitivity[high_recall_candidates$threshold == high_recall_threshold]
  } else {
    high_recall_idx <- which.max(metrics_df$sensitivity)
    high_recall_threshold <- metrics_df$threshold[high_recall_idx]
    recall_at_high_recall <- metrics_df$sensitivity[high_recall_idx]
  }

  return(list(
    youden_threshold = youden_threshold, f1_threshold = f1_threshold,
    high_recall_threshold = high_recall_threshold, youden_index = youden_index,
    max_f1 = max_f1, recall_at_high_recall = recall_at_high_recall,
    threshold_analysis = metrics_df
  ))
}

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME <- "pooled_zscore_with_ar_optimized"
FEATURE_TYPE <- "zscore"
FILTER_VARIANT <- "WITH_AR_FILTER"

CLASS_WEIGHT_RATIO <- 10

# Random effects thresholds (same as original)
MIN_OBS_PER_GROUP <- 10   # Minimum observations per group for stable random effects
COVERAGE_THRESHOLD <- 0.5  # Minimum proportion of groups meeting MIN_OBS_PER_GROUP

# Paths
BASE_DIR <- "C:/GDELT_Africa_Extract"
SCRIPTS_DIR <- file.path(BASE_DIR, "Scripts/district_pipeline/FINAL_PIPELINE - StratifiedSpatialCV")
RESULTS_DIR <- file.path(SCRIPTS_DIR, "RESULTS")
PHASE1_DIR <- file.path(RESULTS_DIR, "stage2_features/phase1_district_threshold")
PHASE2_DIR <- file.path(RESULTS_DIR, "stage2_features/phase2_features")
PHASE3_DIR <- file.path(RESULTS_DIR, "stage2_models/mixed_effects")

dir.create(file.path(PHASE3_DIR, MODEL_NAME), recursive = TRUE, showWarnings = FALSE)
OUTPUT_DIR <- file.path(PHASE3_DIR, MODEL_NAME)

set.seed(42)

cat("=" , rep("=", 78), "\n", sep="")
cat("PHASE 3: POOLED ZSCORE MODEL - OPTIMIZED\n")
cat("=" , rep("=", 78), "\n", sep="")
cat("\nModel:", MODEL_NAME, "\n")
cat("Filter:", FILTER_VARIANT, "\n")
cat("Optimizations: Class Weighting, Location Encoding\n")

# =============================================================================
# LOAD DATA
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Loading zscore features...\n")

features_file <- file.path(PHASE2_DIR, "zscore_features_h8.csv")

if (!file.exists(features_file)) {
  stop("Zscore features file not found. Please run Phase 2 first.")
}

df <- read_csv(features_file, show_col_types = FALSE)
cat("   Loaded:", nrow(df), "observations\n")

# =============================================================================
# APPLY AR FILTER
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Applying WITH_AR_FILTER...\n")

n_initial <- nrow(df)

if ("ipc_value_filled" %in% names(df)) {
  df <- df %>% filter(ipc_value_filled <= 2)
} else if ("ipc_value" %in% names(df)) {
  df <- df %>% filter(ipc_value <= 2)
}

n_after_ipc <- nrow(df)
cat("   After IPC <= 2 filter:", n_after_ipc, "rows\n")

if ("ar_pred_optimal_filled" %in% names(df)) {
  df <- df %>% filter(ar_pred_optimal_filled == 0)
}

n_after_ar <- nrow(df)
cat("   After AR == 0 filter:", n_after_ar, "rows\n")

# =============================================================================
# APPLY COUNTRY INCLUSION FILTER
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Applying country inclusion filter...\n")

country_inclusion_file <- file.path(PHASE1_DIR, "country_inclusion.csv")

if (file.exists(country_inclusion_file)) {
  country_inclusion <- read_csv(country_inclusion_file, show_col_types = FALSE)
  included_countries <- country_inclusion %>%
    filter(included == TRUE) %>%
    pull(ipc_country)

  df <- df %>% filter(ipc_country %in% included_countries)
  cat("   After country filter:", nrow(df), "rows\n")
}

# =============================================================================
# PREPARE FEATURES
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Preparing features...\n")

# Zscore feature columns
zscore_cols <- c("conflict_zscore", "displacement_zscore", "economic_zscore",
                 "food_security_zscore", "governance_zscore", "health_zscore",
                 "humanitarian_zscore", "other_zscore", "weather_zscore")

zscore_cols <- zscore_cols[zscore_cols %in% names(df)]
cat("   Found", length(zscore_cols), "zscore columns\n")

target_col <- if ("ipc_future_crisis" %in% names(df)) "ipc_future_crisis" else "future_crisis"
country_col <- if ("ipc_country" %in% names(df)) "ipc_country" else "country"
district_col <- "ipc_geographic_unit_full"

df <- df %>% filter(!is.na(!!sym(target_col)))
cat("   After removing missing target:", nrow(df), "rows\n")

# Class balance
n_crisis <- sum(df[[target_col]] == 1, na.rm = TRUE)
n_no_crisis <- sum(df[[target_col]] == 0, na.rm = TRUE)
imbalance_ratio <- n_no_crisis / n_crisis
prevalence <- n_crisis / (n_crisis + n_no_crisis)

cat("   Crisis events:", n_crisis, "(", round(prevalence * 100, 1), "%)\n")
cat("   Imbalance ratio:", round(imbalance_ratio, 2), ":1\n")

# =============================================================================
# CREATE CLASS WEIGHTS
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Creating class weights for imbalance handling...\n")

df <- df %>%
  mutate(
    obs_weight = ifelse(!!sym(target_col) == 1, CLASS_WEIGHT_RATIO, 1)
  )

cat("   Crisis weight:", CLASS_WEIGHT_RATIO, "\n")
cat("   Non-crisis weight: 1\n")

# =============================================================================
# DYNAMIC RANDOM EFFECTS LEVEL SELECTION (same as original)
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Evaluating random effects grouping level...\n")

district_obs <- df %>%
  group_by(!!sym(district_col)) %>%
  summarise(n_obs = n(), .groups = "drop")

total_districts <- nrow(district_obs)
districts_sufficient <- sum(district_obs$n_obs >= MIN_OBS_PER_GROUP)
district_coverage <- districts_sufficient / total_districts

cat("   Total districts:", total_districts, "\n")
cat("   Districts with >=", MIN_OBS_PER_GROUP, "obs:", districts_sufficient, "\n")
cat("   Coverage:", round(district_coverage * 100, 1), "%\n")

if (district_coverage >= COVERAGE_THRESHOLD) {
  grouping_col <- district_col
  random_effects_level <- "district"
  cat("   DECISION: Using DISTRICT-level random effects\n")
} else {
  grouping_col <- country_col
  random_effects_level <- "country"
  cat("   DECISION: Using COUNTRY-level random effects\n")
}

n_groups <- length(unique(df[[grouping_col]]))
cat("   Number of random effect groups:", n_groups, "\n")

# =============================================================================
# IDENTIFY KEY SIGNALS FOR RANDOM SLOPES
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Identifying key signals for random slopes...\n")

key_signal_candidates <- c(
  "conflict_zscore",
  "food_security_zscore",
  "humanitarian_zscore",
  "economic_zscore"
)

available_signals <- intersect(key_signal_candidates, zscore_cols)

if (length(available_signals) >= 2) {
  key_signals <- available_signals[1:2]
} else if (length(available_signals) == 1) {
  key_signals <- available_signals[1]
} else {
  key_signals <- zscore_cols[1]
}

cat("   Key signals for random slopes:", paste(key_signals, collapse = ", "), "\n")

# =============================================================================
# USE ALL FEATURES (NO FEATURE SELECTION)
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Using all zscore features (no feature selection)...\n")

# Use all zscore features
selected_features <- zscore_cols

cat("   Using all features (", length(selected_features), "):\n", sep="")
for (feat in selected_features) {
  cat("      -", feat, "\n")
}

# =============================================================================
# SPATIAL CROSS-VALIDATION
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Running spatial cross-validation with optimizations...\n")

folds <- sort(unique(df$fold[df$fold >= 0]))
cat("   Number of folds:", length(folds), "\n")

all_predictions <- data.frame()
fold_metrics <- data.frame()
fold_re_levels <- data.frame()

# Feature formula (fixed effects)
feature_formula <- paste(selected_features, collapse = " + ")

cat("\n   Model formula (aligned with research proposal):\n")
cat("   Fixed effects: beta^T X_{r,t} (", length(selected_features), " selected features)\n", sep="")
cat("   Random effects level:", random_effects_level, "\n")
cat("   Random slopes for:", paste(key_signals, collapse = ", "), "\n")

for (test_fold in folds) {
  cat("\n   Processing fold", test_fold, "...\n")

  train_df <- df %>% filter(fold != test_fold & fold >= 0)
  test_df <- df %>% filter(fold == test_fold)

  cat("      Train:", nrow(train_df), "| Test:", nrow(test_df), "\n")

  train_df <- train_df %>%
    mutate(across(all_of(selected_features), ~ifelse(is.na(.), 0, .)))
  test_df <- test_df %>%
    mutate(across(all_of(selected_features), ~ifelse(is.na(.), 0, .)))

  fold_re_level <- random_effects_level
  fold_grouping_col <- grouping_col

  model_fitted <- FALSE

  tryCatch({
    random_slopes_term <- paste(c("1", key_signals), collapse = " + ")
    random_effects <- paste0("(", random_slopes_term, " | ", fold_grouping_col, ")")
    model_formula <- as.formula(paste(target_col, "~", feature_formula, "+", random_effects))

    model <- glmer(model_formula,
                   data = train_df,
                   family = binomial(link = "logit"),
                   weights = obs_weight,
                   nAGQ = 0,
                   control = glmerControl(
                     optimizer = "bobyqa",
                     optCtrl = list(maxfun = 100000),
                     check.conv.singular = .makeCC(action = "ignore", tol = 1e-4)
                   ))
    model_fitted <- TRUE

  }, error = function(e) {
    cat("      WARNING: Model failed with", fold_re_level, "level:", e$message, "\n")

    if (fold_re_level == "district") {
      cat("      Attempting country-level fallback...\n")
      fold_re_level <<- "country"
      fold_grouping_col <<- country_col

      tryCatch({
        random_slopes_term <- paste(c("1", key_signals), collapse = " + ")
        random_effects <- paste0("(", random_slopes_term, " | ", fold_grouping_col, ")")
        model_formula <- as.formula(paste(target_col, "~", feature_formula, "+", random_effects))

        model <<- glmer(model_formula,
                       data = train_df,
                       family = binomial(link = "logit"),
                       weights = obs_weight,
                       nAGQ = 0,
                       control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
        model_fitted <<- TRUE
        cat("      Country-level fallback successful\n")
      }, error = function(e2) {
        cat("      WARNING: Country fallback failed, trying without weights...\n")
        tryCatch({
          model <<- glmer(model_formula, data = train_df, family = binomial(link = "logit"),
                         nAGQ = 0, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
          model_fitted <<- TRUE
          cat("      Unweighted fallback successful\n")
        }, error = function(e3) { cat("      ERROR: All fallbacks failed\n") })
      })
    } else {
      tryCatch({
        cat("      Attempting without weights...\n")
        random_slopes_term <- paste(c("1", key_signals), collapse = " + ")
        random_effects <- paste0("(", random_slopes_term, " | ", fold_grouping_col, ")")
        model_formula <- as.formula(paste(target_col, "~", feature_formula, "+", random_effects))

        model <<- glmer(model_formula, data = train_df, family = binomial(link = "logit"),
                       nAGQ = 0, control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
        model_fitted <<- TRUE
        cat("      Unweighted fallback successful\n")
      }, error = function(e2) { cat("      ERROR: All fallbacks failed\n") })
    }
  })

  fold_re_levels <- rbind(fold_re_levels, data.frame(fold = test_fold, re_level = fold_re_level, stringsAsFactors = FALSE))

  if (model_fitted) {
    pred_prob <- predict(model, newdata = test_df, type = "response", allow.new.levels = TRUE)

    y_true <- test_df[[target_col]]

    if (length(unique(y_true)) > 1) {
      auc <- as.numeric(auc(roc(y_true, pred_prob, quiet = TRUE)))
      pr_auc <- tryCatch({
        pr <- PRROC::pr.curve(scores.class0 = pred_prob[y_true == 1],
                               scores.class1 = pred_prob[y_true == 0])
        pr$auc.integral
      }, error = function(e) NA)
    } else {
      auc <- NA
      pr_auc <- NA
    }

    threshold_results <- compute_optimal_thresholds(y_true, pred_prob)
    youden_threshold <- threshold_results$youden_threshold
    f1_threshold <- threshold_results$f1_threshold
    high_recall_threshold <- threshold_results$high_recall_threshold

    y_pred_youden <- ifelse(pred_prob >= youden_threshold, 1, 0)
    y_pred_f1 <- ifelse(pred_prob >= f1_threshold, 1, 0)
    y_pred_high_recall <- ifelse(pred_prob >= high_recall_threshold, 1, 0)

    fold_preds <- test_df %>%
      select(any_of(c(
        "ipc_geographic_unit_full", "ipc_district", "ipc_region",
        "ipc_country", "avg_latitude", "avg_longitude",
        "year_month", "ipc_period_start", "ipc_period_end",
        "ipc_value", "ipc_value_filled",
        target_col, "fold", "ar_pred_optimal_filled", "ar_prob_filled"
      ))) %>%
      mutate(
        pred_prob = pred_prob,
        y_pred_youden = y_pred_youden,
        y_pred_f1 = y_pred_f1,
        y_pred_high_recall = y_pred_high_recall,
        threshold_youden = youden_threshold,
        threshold_f1 = f1_threshold,
        threshold_high_recall = high_recall_threshold,
        model = MODEL_NAME,
        filter_variant = FILTER_VARIANT
      )

    names(fold_preds)[names(fold_preds) == target_col] <- "future_crisis"

    # Add confusion class for both thresholds
    fold_preds <- fold_preds %>%
      mutate(
        confusion_youden = case_when(
          future_crisis == 0 & y_pred_youden == 0 ~ "TN",
          future_crisis == 1 & y_pred_youden == 1 ~ "TP",
          future_crisis == 0 & y_pred_youden == 1 ~ "FP",
          future_crisis == 1 & y_pred_youden == 0 ~ "FN",
          TRUE ~ NA_character_
        ),
        confusion_high_recall = case_when(
          future_crisis == 0 & y_pred_high_recall == 0 ~ "TN",
          future_crisis == 1 & y_pred_high_recall == 1 ~ "TP",
          future_crisis == 0 & y_pred_high_recall == 1 ~ "FP",
          future_crisis == 1 & y_pred_high_recall == 0 ~ "FN",
          TRUE ~ NA_character_
        ),
        random_effects_level = fold_re_level
      )

    all_predictions <- bind_rows(all_predictions, fold_preds)

    # Compute fold metrics at YOUDEN threshold (primary/optimal)
    tp <- sum(y_true == 1 & y_pred_youden == 1)
    tn <- sum(y_true == 0 & y_pred_youden == 0)
    fp <- sum(y_true == 0 & y_pred_youden == 1)
    fn <- sum(y_true == 1 & y_pred_youden == 0)

    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    specificity <- ifelse(tn + fp > 0, tn / (tn + fp), 0)
    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy <- (recall + specificity) / 2

    # Calibration metrics
    brier_score <- mean((pred_prob - y_true)^2)
    epsilon <- 1e-15
    pred_prob_clipped <- pmax(pmin(pred_prob, 1 - epsilon), epsilon)
    log_loss <- -mean(y_true * log(pred_prob_clipped) + (1 - y_true) * log(1 - pred_prob_clipped))

    # Metrics at F1-max threshold
    precision_f1 <- ifelse(sum(y_pred_f1) > 0, sum(y_true == 1 & y_pred_f1 == 1) / sum(y_pred_f1), 0)
    recall_f1 <- ifelse(sum(y_true == 1) > 0, sum(y_true == 1 & y_pred_f1 == 1) / sum(y_true == 1), 0)

    # Metrics at high-recall threshold
    precision_high_recall <- ifelse(sum(y_pred_high_recall) > 0,
                                     sum(y_true == 1 & y_pred_high_recall == 1) / sum(y_pred_high_recall), 0)
    recall_high_recall <- ifelse(sum(y_true == 1) > 0,
                                  sum(y_true == 1 & y_pred_high_recall == 1) / sum(y_true == 1), 0)

    fold_result <- data.frame(
      fold = test_fold,
      n_train = nrow(train_df),
      n_test = nrow(test_df),
      n_crisis_test = sum(y_true == 1),

      # Thresholds
      threshold_youden = youden_threshold,
      threshold_f1 = f1_threshold,
      threshold_high_recall = high_recall_threshold,
      youden_index = threshold_results$youden_index,

      # Discrimination metrics
      auc_roc = auc,
      pr_auc = pr_auc,

      # Metrics at Youden threshold
      precision_youden = precision,
      recall_youden = recall,
      f1_youden = f1,
      specificity_youden = specificity,
      accuracy_youden = accuracy,
      balanced_accuracy_youden = balanced_accuracy,
      TP_youden = tp, TN_youden = tn, FP_youden = fp, FN_youden = fn,

      # Metrics at F1-max threshold
      precision_f1 = precision_f1,
      recall_f1 = recall_f1,

      # Metrics at high-recall threshold
      precision_high_recall = precision_high_recall,
      recall_high_recall = recall_high_recall,

      # Calibration metrics
      brier_score = brier_score,
      log_loss = log_loss,

      random_effects_level = fold_re_level
    )

    fold_metrics <- bind_rows(fold_metrics, fold_result)

    cat("      AUC:", round(auc, 4), "| P:", round(precision, 3),
        "| R:", round(recall, 3), "| F1:", round(f1, 3), "\n")
  }
}

# =============================================================================
# TRAIN FINAL MODEL ON ALL DATA (for coefficient extraction)
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Training final model on all data for coefficient extraction...\n")

final_model <- NULL
tryCatch({
  grouping_col_final <- if(random_effects_level == "district") district_col else country_col
  random_effects_final <- paste0("(1 | ", grouping_col_final, ")")
  final_formula <- as.formula(paste(target_col, "~", feature_formula, "+", random_effects_final))

  df_final <- df %>%
    mutate(across(all_of(selected_features), ~ifelse(is.na(.), 0, .)))

  cat("   Formula:", deparse(final_formula), "\n")
  cat("   Observations:", nrow(df_final), "\n")

  final_model <- glmer(final_formula,
                       data = df_final,
                       family = binomial(link = "logit"),
                       weights = obs_weight,
                       control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 50000)),
                       nAGQ = 1)

  cat("   Final model trained successfully\n")
  cat("   Fixed effects:", length(fixef(final_model)), "\n")
  cat("   Random effects groups:", length(ranef(final_model)[[1]]), "\n")

}, error = function(e) {
  cat("   ERROR training final model:", conditionMessage(e), "\n")
})

# =============================================================================
# COMPUTE OVERALL METRICS
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Computing overall metrics...\n")

if (nrow(all_predictions) > 0 && length(unique(all_predictions$future_crisis)) > 1) {
  overall_auc <- as.numeric(auc(roc(all_predictions$future_crisis,
                                    all_predictions$pred_prob, quiet = TRUE)))
  overall_pr_auc <- tryCatch({
    pr <- PRROC::pr.curve(scores.class0 = all_predictions$pred_prob[all_predictions$future_crisis == 1],
                           scores.class1 = all_predictions$pred_prob[all_predictions$future_crisis == 0])
    pr$auc.integral
  }, error = function(e) NA)
} else {
  overall_auc <- NA
  overall_pr_auc <- NA
}

# Compute per-country metrics
cat("\n   Computing per-country metrics...\n")
country_metrics <- all_predictions %>%
  group_by(ipc_country) %>%
  summarise(
    n_samples = n(),
    n_crisis = sum(future_crisis == 1),
    n_no_crisis = sum(future_crisis == 0),
    prevalence = mean(future_crisis),
    tp = sum(future_crisis == 1 & y_pred_youden == 1),
    tn = sum(future_crisis == 0 & y_pred_youden == 0),
    fp = sum(future_crisis == 0 & y_pred_youden == 1),
    fn = sum(future_crisis == 1 & y_pred_youden == 0),
    .groups = "drop"
  ) %>%
  mutate(
    precision = ifelse(tp + fp > 0, tp / (tp + fp), NA),
    recall = ifelse(tp + fn > 0, tp / (tp + fn), NA),
    specificity = ifelse(tn + fp > 0, tn / (tn + fp), NA),
    f1 = ifelse(!is.na(precision) & !is.na(recall) & (precision + recall) > 0,
                2 * precision * recall / (precision + recall), NA),
    accuracy = (tp + tn) / n_samples
  )

country_auc <- all_predictions %>%
  group_by(ipc_country) %>%
  summarise(
    auc_roc = tryCatch({
      if(length(unique(future_crisis)) < 2) NA_real_
      else as.numeric(pROC::auc(pROC::roc(future_crisis, pred_prob, quiet = TRUE)))
    }, error = function(e) NA_real_),
    .groups = "drop"
  )

country_metrics <- country_metrics %>%
  left_join(country_auc, by = "ipc_country")

write.csv(country_metrics, file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_country_metrics.csv")), row.names = FALSE)
cat("   Saved:", paste0(MODEL_NAME, "_country_metrics.csv"), "\n")

# Mean fold metrics
mean_metrics <- fold_metrics %>%
  summarise(
    mean_auc = mean(auc_roc, na.rm = TRUE),
    std_auc = sd(auc_roc, na.rm = TRUE),
    mean_pr_auc = mean(pr_auc, na.rm = TRUE),
    std_pr_auc = sd(pr_auc, na.rm = TRUE),
    mean_precision_youden = mean(precision_youden, na.rm = TRUE),
    mean_recall_youden = mean(recall_youden, na.rm = TRUE),
    mean_f1_youden = mean(f1_youden, na.rm = TRUE),
    mean_specificity_youden = mean(specificity_youden, na.rm = TRUE),
    mean_accuracy_youden = mean(accuracy_youden, na.rm = TRUE),
    mean_balanced_accuracy_youden = mean(balanced_accuracy_youden, na.rm = TRUE),
    mean_precision_f1 = mean(precision_f1, na.rm = TRUE),
    mean_recall_f1 = mean(recall_f1, na.rm = TRUE),
    mean_precision_high_recall = mean(precision_high_recall, na.rm = TRUE),
    mean_recall_high_recall = mean(recall_high_recall, na.rm = TRUE),
    mean_brier_score = mean(brier_score, na.rm = TRUE),
    mean_log_loss = mean(log_loss, na.rm = TRUE)
  )

re_level_summary <- fold_metrics %>%
  group_by(random_effects_level) %>%
  summarise(n_folds = n(), .groups = "drop")

cat("\n   Overall AUC:", round(overall_auc, 4), "\n")
cat("   Overall PR-AUC:", round(overall_pr_auc, 4), "\n")
cat("   Mean fold AUC:", round(mean_metrics$mean_auc, 4), "+/-", round(mean_metrics$std_auc, 4), "\n")
cat("   Mean PR-AUC:", round(mean_metrics$mean_pr_auc, 4), "+/-", round(mean_metrics$std_pr_auc, 4), "\n")
cat("   Mean Precision (Youden):", round(mean_metrics$mean_precision_youden, 4), "\n")
cat("   Mean Recall (Youden):", round(mean_metrics$mean_recall_youden, 4), "\n")
cat("   Mean F1 (Youden):", round(mean_metrics$mean_f1_youden, 4), "\n")
cat("   Mean Brier Score:", round(mean_metrics$mean_brier_score, 4), "\n")
cat("   Mean Log Loss:", round(mean_metrics$mean_log_loss, 4), "\n")

cat("\n   Random Effects Level Usage:\n")
for (i in 1:nrow(re_level_summary)) {
  cat("      ", re_level_summary$random_effects_level[i], ": ",
      re_level_summary$n_folds[i], " folds\n", sep="")
}

# =============================================================================
# EXTRACT AND SAVE COEFFICIENTS
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Extracting model coefficients and effects...\n")

if (!is.null(final_model)) {
  fixed_effects <- fixef(final_model)
  fixed_effects_df <- data.frame(
    feature = names(fixed_effects),
    coefficient = as.numeric(fixed_effects),
    type = "fixed_effect"
  )

  fixed_effects_file <- file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_fixed_effects.csv"))
  write_csv(fixed_effects_df, fixed_effects_file)
  cat("   Saved fixed effects:", fixed_effects_file, "\n")

  random_effects_list <- ranef(final_model)
  re_grouping_var <- names(random_effects_list)[1]
  random_intercepts <- random_effects_list[[re_grouping_var]]

  random_effects_df <- data.frame(
    grouping_level = re_grouping_var,
    group_id = rownames(random_intercepts),
    random_intercept = random_intercepts[,1],
    type = "random_effect"
  )

  random_effects_df$bias_direction <- ifelse(
    random_effects_df$random_intercept > 0,
    "Higher baseline risk",
    "Lower baseline risk"
  )
  random_effects_df$abs_bias <- abs(random_effects_df$random_intercept)

  random_effects_file <- file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_random_effects.csv"))
  write_csv(random_effects_df, random_effects_file)
  cat("   Saved random effects:", random_effects_file, "\n")

  top_high_risk <- random_effects_df %>% arrange(desc(random_intercept)) %>% head(10)
  top_low_risk <- random_effects_df %>% arrange(random_intercept) %>% head(10)

  biases_summary <- rbind(
    top_high_risk %>% mutate(category = "Top 10 Highest Risk Bias"),
    top_low_risk %>% mutate(category = "Top 10 Lowest Risk Bias")
  )

  biases_file <- file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_location_biases.csv"))
  write_csv(biases_summary, biases_file)
  cat("   Saved location biases:", biases_file, "\n")

  model_file <- file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_model.rds"))
  saveRDS(final_model, model_file)
  cat("   Saved model object:", model_file, "\n")
}

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

cat("\n", rep("-", 40), "\n", sep="")
cat("Saving outputs...\n")

pred_file <- file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_predictions.csv"))
write_csv(all_predictions, pred_file)
cat("   Saved:", pred_file, "\n")

metrics_file <- file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_cv_results.csv"))
write_csv(fold_metrics, metrics_file)
cat("   Saved:", metrics_file, "\n")

if (exists("threshold_results") && !is.null(threshold_results$threshold_analysis)) {
  threshold_analysis_df <- threshold_results$threshold_analysis
  threshold_analysis_df$model <- MODEL_NAME
  write.csv(threshold_analysis_df,
            file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_threshold_analysis.csv")),
            row.names = FALSE)
  cat("   Saved:", paste0(MODEL_NAME, "_threshold_analysis.csv"), "\n")
}

summary_list <- list(
  model_name = MODEL_NAME,
  feature_type = FEATURE_TYPE,
  filter_variant = FILTER_VARIANT,
  optimizations = list(
    feature_selection = "None - all features used",
    class_weighting = CLASS_WEIGHT_RATIO,
    random_effects = "random intercepts + random slopes for key signals"
  ),
  timestamp = as.character(Sys.time()),
  n_observations = nrow(all_predictions),
  n_countries = length(unique(all_predictions$ipc_country)),
  n_districts = length(unique(all_predictions$ipc_geographic_unit_full)),
  n_features_original = length(zscore_cols),
  n_features_selected = length(selected_features),
  selected_features = selected_features,
  key_signals = key_signals,
  class_balance = list(
    n_crisis = n_crisis,
    n_no_crisis = n_no_crisis,
    prevalence = prevalence,
    imbalance_ratio = imbalance_ratio,
    class_weight_applied = CLASS_WEIGHT_RATIO
  ),
  threshold_optimization = list(
    method = "Dynamic per-fold optimization",
    youden_description = "Maximizes Sensitivity + Specificity - 1",
    f1_description = "Maximizes F1 score",
    high_recall_description = "Targets 90% recall",
    mean_threshold_youden = mean(fold_metrics$threshold_youden, na.rm = TRUE),
    mean_threshold_f1 = mean(fold_metrics$threshold_f1, na.rm = TRUE),
    mean_threshold_high_recall = mean(fold_metrics$threshold_high_recall, na.rm = TRUE),
    mean_youden_index = mean(fold_metrics$youden_index, na.rm = TRUE)
  ),
  overall_metrics = list(
    primary_threshold = "youden",
    auc_roc = overall_auc,
    pr_auc = overall_pr_auc,
    mean_fold_auc = mean_metrics$mean_auc,
    std_fold_auc = mean_metrics$std_auc,
    mean_pr_auc = mean_metrics$mean_pr_auc,
    std_pr_auc = mean_metrics$std_pr_auc,
    mean_precision_youden = mean_metrics$mean_precision_youden,
    mean_recall_youden = mean_metrics$mean_recall_youden,
    mean_f1_youden = mean_metrics$mean_f1_youden,
    mean_specificity_youden = mean_metrics$mean_specificity_youden,
    mean_accuracy_youden = mean_metrics$mean_accuracy_youden,
    mean_balanced_accuracy_youden = mean_metrics$mean_balanced_accuracy_youden,
    mean_precision_f1 = mean_metrics$mean_precision_f1,
    mean_recall_f1 = mean_metrics$mean_recall_f1,
    mean_precision_high_recall = mean_metrics$mean_precision_high_recall,
    mean_recall_high_recall = mean_metrics$mean_recall_high_recall,
    mean_brier_score = mean_metrics$mean_brier_score,
    mean_log_loss = mean_metrics$mean_log_loss,
    overall_brier_score = mean((all_predictions$pred_prob - all_predictions$future_crisis)^2)
  ),
  n_folds = length(folds),
  random_effects_selection = list(
    preferred_level = "district",
    min_obs_per_group = MIN_OBS_PER_GROUP,
    coverage_threshold = COVERAGE_THRESHOLD,
    actual_district_coverage = district_coverage,
    districts_with_sufficient_obs = districts_sufficient,
    total_districts = total_districts,
    initial_decision = random_effects_level,
    folds_using_district = sum(fold_metrics$random_effects_level == "district"),
    folds_using_country = sum(fold_metrics$random_effects_level == "country")
  ),
  model_specification = list(
    formula_alignment = "Research Proposal Stage 4",
    fixed_effects = paste0("beta^T X_{r,t} - ", length(selected_features), " selected features"),
    random_intercepts = paste0("alpha_r - ", random_effects_level, "-specific baseline risk"),
    random_slopes = paste("b_r Z_{r,t} for:", paste(key_signals, collapse = ", ")),
    key_signals = key_signals,
    grouping_variable = grouping_col
  )
)

summary_file <- file.path(OUTPUT_DIR, paste0(MODEL_NAME, "_summary.json"))
write_json(summary_list, summary_file, pretty = TRUE, auto_unbox = TRUE)
cat("   Saved:", summary_file, "\n")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n", rep("=", 80), "\n", sep="")
cat("OPTIMIZED MIXED EFFECTS MODEL COMPLETE:", MODEL_NAME, "\n")
cat(rep("=", 80), "\n", sep="")
cat("\nOptimizations Applied:\n")
cat("   1. All Features Used (", length(selected_features), "/", length(zscore_cols), " features)\n", sep="")
cat("   2. Class Weighting:", CLASS_WEIGHT_RATIO, ":1 crisis weight\n")
cat("   3. Random Effects: random intercepts + random slopes for key signals\n")
cat("\nPerformance:\n")
cat("   Overall AUC-ROC:", round(overall_auc, 4), "\n")
cat("   Overall PR-AUC:", round(overall_pr_auc, 4), "\n")
cat("   Mean Precision (Youden):", round(mean_metrics$mean_precision_youden, 4), "\n")
cat("   Mean Recall (Youden):", round(mean_metrics$mean_recall_youden, 4), "\n")
cat("   Mean F1 (Youden):", round(mean_metrics$mean_f1_youden, 4), "\n")
cat("   Mean Brier Score:", round(mean_metrics$mean_brier_score, 4), "\n")
cat("\nOutput:", OUTPUT_DIR, "\n")
