# Metrics, cross validation, etc.
from sksurv.metrics import concordance_index_ipcw
from ..config import TAU


def evaluate_model_cindex(model, X_train, y_train, X_test, y_test, tau=TAU):
    """Evaluate a model with C-index IPCW"""
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # C-index IPCW
    train_cindex = concordance_index_ipcw(y_train, y_train, train_pred, tau=tau)[0]
    test_cindex = concordance_index_ipcw(y_train, y_test, test_pred, tau=tau)[0]
    results = {
        "train_cindex": train_cindex,
        "test_cindex": test_cindex,
        "train_predictions": train_pred,
        "test_predictions": test_pred,
    }
    print(f"C-Index IPCW Train: {results['train_cindex']:.5f}")
    print(f"C-Index IPCW Test: {results['test_cindex']:.5f}")

    return results


def compare_models(models, X_train, y_train, X_test, y_test):
    """Compare multiple models"""
    results = {}

    for name, model_info in models.items():
        model = model_info["model"]
        print(f"\nEvaluating model {name}...")

        eval_results = evaluate_model_cindex(model, X_train, y_train, X_test, y_test)

        print(f"{name} - C-Index IPCW Train: {eval_results['train_cindex']:.5f}")
        print(f"{name} - C-Index IPCW Test: {eval_results['test_cindex']:.5f}")

        results[name] = eval_results

    # Find the best model
    best_model = max(results.keys(), key=lambda k: results[k]["test_cindex"])
    print(
        f"\nBest model: {best_model} (C-Index Test: {results[best_model]['test_cindex']:.5f})"
    )

    return results, best_model
