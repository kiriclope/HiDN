  import numpy as np

  def safe_roc_auc_score(y_true, y_score):
      y_true = np.asarray(y_true)
      if len(np.unique(y_true)) == 1:
          return np.nan  # return np.nan where the score cannot be calculated
      return roc_auc_score(y_true, y_score)

  def rescale_coefs(model, coefs, bias):

          try:
                  means = model.named_steps["scaler"].mean_
                  scales = model.named_steps["scaler"].scale_

                  # Rescale the coefficients
                  rescaled_coefs = np.true_divide(coefs, scales)

                  # Adjust the intercept
                  rescaled_bias = bias - np.sum(rescaled_coefs * means)

                  return rescaled_coefs, rescaled_bias
          except:
                  return coefs, bias

  from scipy.stats import bootstrap

  def get_bootstrap_ci(data, statistic=np.mean, confidence_level=0.95, n_resamples=1000, random_state=None):
      result = bootstrap((data,), statistic)
      ci_lower, ci_upper = result.confidence_interval
      return np.array([ci_lower, ci_upper])

  def convert_seconds(seconds):
      h = seconds // 3600
      m = (seconds % 3600) // 60
      s = seconds % 60
      return h, m, s

  import pickle as pkl

  def pkl_save(obj, name, path="."):
      pkl.dump(obj, open(path + "/" + name + ".pkl", "wb"))


  def pkl_load(name, path="."):
      return pkl.load(open(path + "/" + name, "rb"))
