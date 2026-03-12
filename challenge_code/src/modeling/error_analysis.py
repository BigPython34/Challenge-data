import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def analyze_cv_errors(oof_predictions: pd.DataFrame, y: np.ndarray, output_dir: str):
    """
    Analyze errors from Cross-Validation OOF predictions.
    Focuses on ranking errors (IPCW C-index proxy).
    
    Args:
        oof_predictions: DataFrame (n_samples, n_models) with risk scores.
        y: Structured array with 'status' (bool/int) and 'time' (float).
        output_dir: Directory to save the analysis report.
    """
    print("\n[ANALYSIS] Starting Error Analysis on OOF predictions...")
    
    # Extract Time and Status
    # sksurv structured array usually has 'event' and 'time'
    if y.dtype.names and 'event' in y.dtype.names:
        events = y["event"].astype(bool)
    elif y.dtype.names and 'status' in y.dtype.names:
        events = y["status"].astype(bool)
    else:
        # Fallback if it's a dataframe or something else (unlikely given load_training_dataset_csv)
        raise ValueError(f"Unknown y structure. Fields: {y.dtype.names}")

    times = y["time"]
    
    # We can only reliably assess "Truth" ranking for uncensored patients (events)
    # For censored patients, we can only say "lived at least X".
    # To keep it simple and robust, we'll focus the quantitative error metric on Events.
    # But we will calculate ranks for everyone to see relative positioning.
    
    analysis_df = pd.DataFrame(index=oof_predictions.index)
    analysis_df["Time"] = times
    analysis_df["Event"] = events
    
    # 1. Compute Risk Ranks for all models (0=Low Risk, 1=High Risk)
    # We normalize to [0, 1]
    risk_ranks = oof_predictions.rank(pct=True)
    
    # Add ranks to analysis_df
    for col in oof_predictions.columns:
        analysis_df[f"Rank_{col}"] = risk_ranks[col]
        
    # 2. Compute "Truth Rank" for Events
    # Longest survivor = Rank 1.0 (Low Risk)
    # Shortest survivor = Rank 0.0 (High Risk)
    # Wait, usually Risk is inversely related to Time.
    # High Risk (1.0) -> Short Time.
    # Low Risk (0.0) -> Long Time.
    
    # Let's define Truth Rank such that it correlates with Risk Rank.
    # Shortest Time = 1.0 (High Risk Truth).
    # Longest Time = 0.0 (Low Risk Truth).
    
    # We only compute this for events
    event_mask = events == True
    event_times = times[event_mask]
    
    # Rank data: smallest time gets rank 1 (min rank), largest gets N.
    # We want Smallest Time -> High Rank (1.0).
    # rankdata gives 1 for smallest.
    # So we want to reverse it? No.
    # If Time is small, Risk should be high.
    # rankdata(times): Small time = Rank 1. Large time = Rank N.
    # We want Small time -> 1.0. Large time -> 0.0.
    # So we use rankdata(-times) or just invert.
    # Let's use rankdata(-times): Largest time (negative) gets rank 1 (smallest). Smallest time (least negative) gets rank N.
    # So Smallest Time -> Rank N -> 1.0. Correct.
    
    truth_ranks_events = rankdata(-event_times, method="average") / len(event_times)
    
    analysis_df["Truth_Rank"] = np.nan
    analysis_df.loc[event_mask, "Truth_Rank"] = truth_ranks_events
    
    # 3. Compute Errors (Risk Rank - Truth Rank)
    # Positive Error (>0): Risk Rank > Truth Rank. Predicted High Risk, but Truth was Lower Risk (Lived Longer than expected for that risk). -> PESSIMISTIC
    # Negative Error (<0): Risk Rank < Truth Rank. Predicted Low Risk, but Truth was Higher Risk (Died Sooner than expected). -> OPTIMISTIC
    
    error_cols = []
    for col in oof_predictions.columns:
        # Error only defined for events
        err_col = f"Error_{col}"
        analysis_df[err_col] = analysis_df[f"Rank_{col}"] - analysis_df["Truth_Rank"]
        error_cols.append(err_col)

    # 4. Consensus and Disagreement
    # Mean Risk Rank across models
    analysis_df["Consensus_Risk_Rank"] = risk_ranks.mean(axis=1)
    # Std Dev of Risk Rank (Disagreement)
    analysis_df["Model_Disagreement"] = risk_ranks.std(axis=1)
    
    # Mean Error (Bias)
    analysis_df["Mean_Error"] = analysis_df[error_cols].mean(axis=1)
    # Mean Absolute Error (Difficulty)
    analysis_df["Mean_Abs_Error"] = analysis_df[error_cols].abs().mean(axis=1)
    
    # 5. Save detailed report
    report_path = os.path.join(output_dir, "error_analysis_detailed.csv")
    analysis_df.to_csv(report_path)
    print(f"[ANALYSIS] Detailed error report saved to: {report_path}")
    
    # 6. Identify "Hardest" samples (Top 20 Mean Abs Error)
    hardest = analysis_df[event_mask].nlargest(20, "Mean_Abs_Error")
    hardest_path = os.path.join(output_dir, "error_analysis_hardest_samples.csv")
    hardest.to_csv(hardest_path)
    print(f"[ANALYSIS] Hardest samples saved to: {hardest_path}")
    
    # 7. Identify "Most Controversial" samples (Top 20 Disagreement)
    controversial = analysis_df.nlargest(20, "Model_Disagreement")
    controversial_path = os.path.join(output_dir, "error_analysis_controversial.csv")
    controversial.to_csv(controversial_path)
    print(f"[ANALYSIS] Most controversial samples saved to: {controversial_path}")
    
    # 8. Model Comparison Summary
    # Who is most optimistic/pessimistic?
    summary_rows = []
    for col in oof_predictions.columns:
        errs = analysis_df[f"Error_{col}"].dropna()
        row = {
            "Model": col,
            "Mean_Error (Bias)": errs.mean(),
            "MAE": errs.abs().mean(),
            "Optimistic_Errors (<-0.2)": (errs < -0.2).mean(),
            "Pessimistic_Errors (>0.2)": (errs > 0.2).mean(),
        }
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "error_analysis_model_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[ANALYSIS] Model summary saved to: {summary_path}")

    return analysis_df
