import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plt_calibration_curve(y_true, y_probs, bins=10):
    """
    y_true: Ground truth labels (not one-hot)
    y_probs: The mean probability output from your ensamble
    """
    # bin_preds = np.argmax(y_probs, axis = 1)
    # confidences = np.max(y_probs, axis=1)

    # accuracies = (bin_preds == y_true)

    # bin_boundaries = np.linspace(0,1, bins+1)
    # bin_accs = []
    # bin_confs = []

    # for i in range(bins):
    #     mask = (confidences > bin_boundaries[i]) &(confidences <= bin_boundaries[i+1])
    #     if np.any(mask):
    #         bin_accs.append(np.mean(accuracies[mask]))
    #         bin_confs.append(np.mean(confidences[mask]))

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)

    correct = (predictions == y_true)

    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=10, strategy='quantile')

    plt.plot(prob_pred, prob_true, marker='o', label='Ensemble (Quantile)')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Actual Accuracy')
    plt.title('Reliability Diagram (Quantile Binning)')
    plt.legend()
    plt.grid(True)
    plt.show()

def confidence_prediction(x_input, models, threshold=0.98):
    """
    Only returns a prediction if the ensamble is sufficiently certain.
    """
    probs, conf, var = bayesian_ensamble_predict(models, x_input)
    
    results = []
    for i in range(len(conf)):
        predicted_class = np.argmax(probs[i])
        confidence_score = conf[i]
        if confidence_score >= threshold:
            print(f"Sample {i}: PREDICT {predicted_class} (Confidence: {confidence_score:.4f}) | Disagreement (Var): {var[i]:.6f}")
            results.append(predicted_class)
        else:
            print(f"Sample {i}: REJECT - Escalate to backup system (Confidence: {confidence_score:.4f}) | Disagreement (Var): {var[i]:.6f}")
            results.append(None)
    return results

def get_standardized_preds(model, x, bnn_samples=10):
    is_conv_model = any("conv" in str(layer).lower() for layer in getattr(model, 'layers', []))
    x_input = x
    if is_conv_model and x.ndim == 2:
        x_input = x.reshape(-1, 28, 28, 1)
    if hasattr(model, "predict_proba"):
        print(type(model))
        return model.predict_proba(x)
    elif hasattr(model, "predict"):
        return model.predict(x_input, verbose=0)
    elif hasattr(model, "sample_predict"):
        return model.sample_predict(x, n_samples=50)
    else:
        raise TypeError(f"Model type {type(model)} not recognized by standardized wrapper.")
    
def bayesian_ensamble_predict(models, x_test):
    all_preds = []
    all_variances = []
    for i, m in enumerate(models):
        preds = get_standardized_preds(m, x_test)
        print(f"Model {m} output shape: {preds.shape}")
        if preds.ndim ==3:
            model_var = np.var(preds, axis=0).mean(axis=1)
            model_mean = np.mean(preds, axis=0)
        else:
            entropy = -np.sum(preds * np.log(preds + 1e-10), axis=1)
            model_var = entropy
            model_mean = preds
        all_preds.append(model_mean)
        all_variances.append(model_var)
    
    weighted_mean_probs = dynamc_weighted_ensamble(all_preds, all_variances)


    ensamble_stack = np.stack(all_preds, axis=0)
    variance_prediction = np.var(ensamble_stack, axis = 0).mean(axis=1)
    confidence = np.max(weighted_mean_probs, axis=1)

    return weighted_mean_probs, confidence, variance_prediction

def dynamic_threshold(probs, y_val, conf):
    thresholds = [0.85, 0.88, 0.90, 0.92, 0.95]
    best_threshold = 0.90
    best_accuracy = 0.0
    results_dict = {}
    total_samples = len(y_val)

    for t in thresholds:
        accepted_indices = np.where(conf > t)[0]
        if len(accepted_indices) > 0:
            predicted_classes = np.argmax(probs[accepted_indices], axis=1)
            true_labels = y_val[accepted_indices]

            accuracy = np.mean(predicted_classes == true_labels)
            rejection_rate = (1-(len(accepted_indices)/total_samples)) * 100

            print(f"Threshold {t}: Accuracy {accuracy:.2%}, Rejection Rate {rejection_rate:.2f}%")
            results_dict[t] = {"accuracy": accuracy, "rejection_rate": rejection_rate}

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
        else:
            print(f"Threshold {t}: No samples accepted.")
    print("-" * 30)
    print(f"Best Threshold: {best_threshold} with Accuracy: {best_accuracy:.2%}")
    
    return best_threshold, best_accuracy, results_dict

def dynamc_weighted_ensamble(all_preds, variance_per_model):
    """
    all_preds: list of arrays
    variance_per_model: list of variances
    """
    # Add small epsilon to avoid division by 0
    precisions = [1.0/(v+1e-6) for v in variance_per_model]

    total_precision = sum(precisions)
    dynamic_weights = [p / total_precision for p in precisions]

    weighted_mean = sum(p* w[:, np.newaxis] for p, w in zip(all_preds, dynamic_weights))

    return weighted_mean