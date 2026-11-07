# Chapter 27 exercises

Time-Series Forecasting

Exercises 1-4 use hand calculations, 5-8 use NumPy, and 9-11 design
chronological forecasting evaluations. All eleven have [complete worked
solutions](../../solutions/ch27_time_series/README.md).
Import the functions from `examples.ch27_time_series.model`.

1. **Separate what is known from what will arrive.** A shop makes Friday's forecast at closing time on Thursday. Classify store size, Friday's scheduled opening hours, Thursday's sales, and Friday's observed foot traffic. State which values can enter the forecast.

2. **Keep targets inside their split.** Observations 0--15 are for training and 16--23 are for testing. Use lookback four and horizon three. Write the last training origin and the first test origin. List the target indices at both origins.

3. **Trace overlapping patches.** Use eight values $(4,2,6,1,7,3,8,5)$, length three, and stride two. List the complete unpadded patches, the indices and values shared by neighboring patches, and the number of ordered query--key pairs including self-pairs. Identify any input value that appears in no complete patch.

4. **Count interval hits.** Two one-step forecasts have intervals $[2,6]$ and $[4,7]$, with actual values 6 and 8. Compute empirical coverage and mean width. Use the chapter's convention for a value at an endpoint. Separately, for actual value 10, compute the 0.9-quantile losses of forecasts 8 and 12 and explain why the losses differ.

5. **Inspect rolling windows.** Call `rolling_windows` on values 0 through 9 with lookback three and horizon two. Identify the first and last origins and print the past and future rows at each. Explain why the last target is at index 9.

6. **Forecast beyond a season.** Use history $(2,5,3,6)$, period two, and horizon five. Run `seasonal_naive`. Use `fit_seasonal_drift(history, period)` and pass its result and the same history to `seasonal_drift`. Identify the first prediction that receives two drift increments.

7. **Change what the baseline learned.** Run Listing 27.3 and inspect the horizon MAEs. Import `series_example` from the runnable module in Listing 27.3. For every origin in `rolling.test_origins`, forecast with history up to that origin, period four, horizon three, and drift zero. Compare with the corresponding seasonal-naive forecast.

8. **Compare interval widths.** Convert the point predictions from the two-origin fixture to a NumPy array. Subtract and add half-widths 0, 1, and 2 to form three pairs of interval arrays. Predict which scores will change. Then call `forecast_metrics` with the actual values, point predictions, and both endpoints for each. Compare horizon-wise coverage and width while tracking what happens to MAE. Explain why choosing a width from these same scored outcomes would not establish prospective 80% coverage.

9. **Audit feature availability.** A daily dataset includes a centered seven-day sales average and a weather forecast column replaced by actual weather after each day. Explain what could leak into a forecast made three days ahead. Design features from data available at each origin.

10. **Choose a rolling protocol.** Design a comparison between a fixed model and one refitted once per week. Specify training, validation, and test periods, test origins, available covariates, and when each model can use new observations. Explain how you would compare compute cost.

11. **Evaluate intervals under a shift.** A delivery service sees a change in seasonal demand after opening a new depot. Design a test that compares coverage and width before and after the change for horizons one and seven. Decide how to handle forecast windows that straddle the opening date. Describe how you would set the intervals before scoring the test forecasts.
