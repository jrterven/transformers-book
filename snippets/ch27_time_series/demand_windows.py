def demand_windows(values, issue_indices):
    """Build observed histories and scored targets around midnight UTC issues."""
    issue = np.asarray(issue_indices)[:, None]
    history_index = issue + np.arange(1 - REPORT_LAG - LOOKBACK, 1 - REPORT_LAG)
    target_index = issue + np.arange(1, HORIZON + 1)
    return values[history_index], values[target_index]
