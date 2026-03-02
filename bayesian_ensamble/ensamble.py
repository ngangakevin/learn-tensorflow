import numpy as np
import matplotlib.pyplot as plt

def plt_calibration_curve(y_true, y_probs, bins=10):
    """
    y_true: Ground truth labels (not one-hot)
    y_probs: The mean probability output from your ensamble
    """
    bin_preds = np.argmax(y_probs, axis = 1)
    confidences = np.max(y_probs, axis=1)

    accuracies = (bin_preds == y_true)

    bin_boundaries = np.linspace(0,1, bins+1)
    bin_accs = []
    bin_confs = []

    for i in range(bins):
        mask = (confidences > bin_boundaries[i]) &(confidences <= bin_boundaries[i+1])
        if np.any(mask):
            bin_accs.append(np.mean(accuracies[mask]))
            bin_confs.append(np.mean(confidences[mask]))
    plt.plot([0,1], [0,1], "--", color="gray", label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, "s-", label="Ensambe Calibration")
    plt.ylabel("Actual Accuracy")
    plt.xlabel("Ensamble Confidence Score")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.show()

def confidence_prediction(x_input, models, threshold=0.98):
    """
    Only returns a prediction if the ensamble is sufficiently certain.
    """
    probs, conf, var = bayesian_ensamble_predict(models, x_input)

    for i in range(len(conf)):
        if conf[i] >= threshold:
            print(f"Sample {i}: PREDICT {np.argmax(probs[i])} (Confidence: {conf[i]:.4f})")
        else:
            print(f"Sample {i}: REJECT - Escalate to backup system (Confidence: {conf[i]:.4f})")
    
def get_standardized_preds(model, x, bnn_samples=10):
    x_4d = x.to_numpy().reshape(-1, 28,28,1) if x.ndim != 4 else x
    x_2d = x.to_numpy().reshape(x.shape[0], -1)

    if hasattr(model, "layers"):
        if any("flipout" in layer.name.lower() for layer in model.layers):
            bnn_runs = np.array([model.predict(x_4d, verbose=0) for _ in range(bnn_samples)])
            return bnn_runs
        else:
            return model.predict(x_4d, verbose=0)
    elif "sklearn" in str(type(model)).lower():
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x_2d)
        else:
            preds = model.predict(x_2d)
            one_hot = np.zeros((len(preds), 10))
            for i, p in enumerate(preds):
                one_hot[i, int(p)] = 1.0
            return one_hot
    else:
        print(f"DEBUG: Model Type is {type(model)}")
        raise ValueError(f"Could not route model of type: {type(model)}")  
    
    # if hasattr(model, "predict_proba"):
    #     x_flat = x.to_numpy().reshape(x.shape[0], -1)
    #     return model.predict_proba(x_flat)
    # elif (hasattr(model, "layers") and any("flipout" in layer.name.lower() for layer in model.layers)):
    #     bnn_runs = np.array([model.predict(x_4d, verbose=0) for _ in range(bnn_samples)])
    #     return bnn_runs
    # else:
    #     return model.predict(x_4d)

def bayesian_ensamble_predict(models, x_test):
    all_preds = []
    for m in models:
        preds = get_standardized_preds(m, x_test)
        if preds.ndim ==3:
            preds = np.mean(preds, axis=0)
        if preds.ndim ==2:
            preds = preds.reshape(1, -1)

        all_preds.append(preds)

    ensamble_stack = np.stack(all_preds)
    mean_probabilities = np.mean(ensamble_stack, axis = 0)
    variance_prediction = np.var(ensamble_stack, axis = 0)
    confidence = np.max(mean_probabilities, axis=1)

    return mean_probabilities, confidence, variance_prediction