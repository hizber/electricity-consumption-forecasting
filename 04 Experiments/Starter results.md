# Shared forecasting starter results

## Evaluation decision recorded before scoring

On 10 November 2026, the shared comparison selected the four next-hour models with 24 completed consumption readings and the same eight Helsinki calendar inputs. It also selected the separate fixed-origin GRU with 24 readings, the same calendar inputs for each target, its horizon index, and the target-aligned previous-day reading. The hourly runs are `lstm_h24_verified`, `gru_h24_verified`, `tcn_h24_verified` and `transformer_h24_verified`. The daily run is `gru_daily_h24_v1`. All use seed 42. Use the checkpoint with the lowest validation MAE for each run, with at most 10 epochs and patience 3. Keep rolling summaries and weather out of this comparison. The validation benefit of rolling summaries was small and depended on the metric. Historical weather availability remains unverified.

Freeze all five runs before opening any test or evaluation split. Score the test and three evaluation periods once with the frozen model and preprocessing. Earlier readings may enter the forecast history when available at the origin. Each hourly model must score the same target keys as the others. Compare daily results only on complete fixed-origin profiles. The figures below use the original source units, which the course dictionary does not specify. UTC assignment boundaries and the 00:00 UTC daily schedule remain provisional. The house's calendar features use `Europe/Helsinki`.

The test and evaluation results below were published on 10 November 2026. The group has now been exposed to those results. If subsequent model or feature decisions use them, those same periods no longer provide an untouched assessment of those decisions. Record any such influence in the final report.

## Data version and reproducibility

- Source SHA-256 `9e8334d93bc3f9f1391216f68f52eedb5c9ebd3c2a5856e97db0177cf76f70a0`.
- Training dataset manifest SHA-256 `1ddf0d15b69cb08a719d31b82129ae4c0cfdd7049f7a213cd40857327ff66149` for the local `prepared_no_weather/rebuild_v2`. The manifest records the three generator hashes, all 24 split CSV hashes, counts and UTC boundaries. Use [the setup guide](../02%20Data/Setup.md) to rebuild it with the current code. The manifest hash may differ with a different builder revision or Python version. Compare the split CSVs against [the reference hashes](shared_features/dataset-sha256.txt) to establish byte-identical data.
- The common hourly set uses a 168-hour eligibility warmup even for shorter model inputs. It contains 32,690 fit targets and 8,784 validation targets. The daily set has 1,361 fit profiles and 365 validation profiles. See [coverage and exclusions](../02%20Data/Quality%20review.md).
- Train with default architecture sizes, CPU, Adam at 0.001 and scaled MSE. Batch size is 128 hourly examples or 32 daily origins. Fit scaling and imputation on the fit split. Use validation for early stopping and compare predictions in original source units. The run JSON files record column names, exact data and code hashes, Python, NumPy and PyTorch versions, and epoch results. See the [model guide](models/README.md).

## Hourly feature experiments on validation

All rows below use the same 8,784 target keys and labels. History is ordered from oldest to newest. The calendar group is `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`, `year_sin`, `year_cos`, `is_weekend` and `is_finnish_public_holiday`. Rolling features are the mean, population standard deviation, minimum, maximum and change over 3, 6 and 24 completed readings in the 24-hour experiments. The 72- and 168-hour comparisons use history and calendar only.

| GRU features | Hours | Best epoch | MAE | RMSE | Training seconds |
|---|---:|---:|---:|---:|---:|
| History | 24 | 5 | 0.6677 | 0.9050 | 33.5 |
| History and rolling | 24 | 10 | 0.6671 | 0.8983 | 23.0 |
| History and calendar | 24 | 5 | 0.6652 | 0.9048 | 16.7 |
| History, calendar and rolling | 24 | 7 | 0.6642 | 0.8996 | 24.4 |
| History and calendar | 72 | 7 | 0.6723 | 0.9067 | 76.6 |
| History and calendar | 168 | 10 | 0.6723 | 0.9077 | 148.7 |
| Previous day | | | 0.9181 | 1.2444 | |

The table uses seed 42 and one validation year. Timings are approximate CPU wall time and depend on concurrent load. In these runs 24 hours gave lower MAE than 72 or 168. Adding calendar and rolling to the 24-hour model lowered MAE by 0.0035 compared with history alone. That difference is too small to establish a stable advantage from one run. Correlation describes association and does not establish feature importance or forecast improvement.

During the 10 November audit, the five non-baseline feature configurations were rerun with seed 42 using the code at commit `5495f62`. Their selected epochs, MAE, RMSE and every hourly and seasonal validation slice reproduced the original values. This repeat checks reproducibility under the same seed; it does not measure variability across seeds. The original runs recorded code hashes for earlier development revisions that were not separately committed.

### Repeated seeds for 24-hour GRU inputs

The audit also trained seeds 1 and 7 with the current split-boundary guard and compared them with the existing seed 42 results. Each configuration used the same 8,784 validation labels. Each cell shows MAE / RMSE at the epoch selected by validation MAE.

| Features | Seed 1 | Seed 7 | Seed 42 | Mean MAE |
|---|---:|---:|---:|---:|
| History | 0.6667 / 0.8975 | 0.6694 / 0.8979 | 0.6677 / 0.9050 | 0.6679 |
| History and rolling | 0.6658 / 0.8993 | 0.6659 / 0.9030 | 0.6671 / 0.8983 | 0.6663 |
| History and calendar | 0.6647 / 0.8915 | 0.6663 / 0.8950 | 0.6652 / 0.9048 | 0.6654 |
| History, calendar and rolling | 0.6613 / 0.8978 | 0.6608 / 0.8886 | 0.6642 / 0.8996 | 0.6621 |
| Calendar and temperature, conditional | 0.6517 / 0.8811 | 0.6569 / 0.8914 | 0.6533 / 0.8833 | 0.6540 |
| Calendar and expanded weather, conditional | 0.6523 / 0.8805 | 0.6589 / 0.8975 | 0.6540 / 0.8826 | 0.6551 |

Calendar and rolling together improved MAE relative to history alone for each of these seeds. Temperature improved MAE over calendar alone in each run under the unverified delay assumption. The expanded weather group had slightly higher MAE than the temperature group for all three seeds. The comparison uses one validation year and a single household. The new seed runs were performed after the starter holdout results had been published.

### Seasonal and hourly validation errors

Season and hour refer to target end time in Helsinki. Each hour uses all eligible dates at that hour.

| Features | Autumn MAE | Winter MAE | Spring MAE | Summer MAE | Worst local hour | MAE at that hour |
|---|---:|---:|---:|---:|---:|---:|
| History | 0.7217 | 0.8266 | 0.7364 | 0.3884 | 20:00 | 1.2355 |
| History and rolling | 0.7137 | 0.8179 | 0.7369 | 0.4022 | 20:00 | 1.2081 |
| History and calendar | 0.7224 | 0.8212 | 0.7413 | 0.3782 | 20:00 | 1.2323 |
| History, calendar and rolling | 0.7123 | 0.8248 | 0.7384 | 0.3834 | 20:00 | 1.1955 |
| History and calendar, 72h | 0.7259 | 0.8240 | 0.7438 | 0.3977 | 20:00 | 1.2175 |
| History and calendar, 168h | 0.7256 | 0.8218 | 0.7414 | 0.4025 | 20:00 | 1.2142 |

| Helsinki target hour | History MAE | History + rolling | History + calendar | All three | Previous day |
|---|---:|---:|---:|---:|---:|
| 00:00 | 0.5730 | 0.5964 | 0.5534 | 0.5776 | 0.6779 |
| 01:00 | 0.5141 | 0.5051 | 0.4989 | 0.4934 | 0.6790 |
| 02:00 | 0.5102 | 0.5009 | 0.4981 | 0.4862 | 0.7118 |
| 03:00 | 0.5127 | 0.5000 | 0.5021 | 0.4875 | 0.7357 |
| 04:00 | 0.4840 | 0.4754 | 0.4735 | 0.4672 | 0.6430 |
| 05:00 | 0.5219 | 0.5051 | 0.5097 | 0.5003 | 0.7425 |
| 06:00 | 0.5226 | 0.5134 | 0.5389 | 0.5219 | 0.7456 |
| 07:00 | 0.6211 | 0.6173 | 0.6322 | 0.6143 | 0.8567 |
| 08:00 | 0.6280 | 0.6097 | 0.6543 | 0.6386 | 0.9360 |
| 09:00 | 0.6262 | 0.6161 | 0.6387 | 0.6285 | 0.8945 |
| 10:00 | 0.6631 | 0.6627 | 0.6756 | 0.6811 | 0.9323 |
| 11:00 | 0.6254 | 0.6291 | 0.6253 | 0.6236 | 0.9286 |
| 12:00 | 0.6453 | 0.6461 | 0.6413 | 0.6483 | 0.9720 |
| 13:00 | 0.6776 | 0.6849 | 0.6777 | 0.6872 | 0.9611 |
| 14:00 | 0.6464 | 0.6551 | 0.6515 | 0.6609 | 0.9505 |
| 15:00 | 0.6006 | 0.6035 | 0.6031 | 0.6042 | 0.9369 |
| 16:00 | 0.7316 | 0.7253 | 0.7157 | 0.7144 | 1.0014 |
| 17:00 | 0.6893 | 0.6913 | 0.6900 | 0.6894 | 0.9299 |
| 18:00 | 0.6739 | 0.6808 | 0.6827 | 0.6906 | 0.9946 |
| 19:00 | 0.8285 | 0.8165 | 0.8217 | 0.8124 | 1.0897 |
| 20:00 | 1.2355 | 1.2081 | 1.2323 | 1.1955 | 1.4654 |
| 21:00 | 1.0640 | 1.0487 | 1.0635 | 1.0504 | 1.3109 |
| 22:00 | 0.7690 | 0.8198 | 0.7460 | 0.7945 | 1.0680 |
| 23:00 | 0.6608 | 0.6994 | 0.6383 | 0.6719 | 0.8712 |

## Conditional hourly weather comparison

The weather features come from the original course CSV's observations. A one-hour publication delay is assumed, so for a target ending at `t + 1` the latest weather observation ends at `t − 1`. The conditional build has the same fit and validation keys and labels as the no-weather build. The temperature group includes latest temperature, a missingness flag and heating degrees with a 15 °C threshold. The expanded group adds 3-, 6- and 24-hour temperature mean and change, latest humidity and wind speed, and their missingness flags. The [provenance note](../02%20Data/Weather%20provenance.md) records the unresolved availability and revision questions.

| GRU with 24-hour history and calendar | MAE | RMSE | Autumn MAE | Winter MAE | Spring MAE | Summer MAE |
|---|---:|---:|---:|---:|---:|---:|
| No weather | 0.6652 | 0.9048 | 0.7224 | 0.8212 | 0.7413 | 0.3782 |
| Temperature group | 0.6533 | 0.8833 | 0.7045 | 0.8012 | 0.7170 | 0.3925 |
| Expanded weather group | 0.6540 | 0.8826 | 0.7024 | 0.8023 | 0.7190 | 0.3946 |

| Missing before fit-only mean imputation | Fit | Validation |
|---|---:|---:|
| Latest temperature and heating degrees, each | 57 | 46 |
| Latest humidity | 138 | 46 |
| Latest wind speed | 439 | 70 |
| Temperature mean and change, each, 3-hour window | 120 | 131 |
| Temperature mean and change, each, 6-hour window | 211 | 251 |
| Temperature mean and change, each, 24-hour window | 715 | 882 |

Temperature improved overall MAE by about 1.8% under the timing assumption. Summer MAE increased in both weather runs. The small RMSE difference between the two weather groups does not isolate the effects of humidity, wind or the added rolling terms.

## Daily validation at one origin

| Forecast | Hourly MAE | Hourly RMSE | 24-hour total MAE | 24-hour total RMSE |
|---|---:|---:|---:|---:|
| GRU | 0.7246 | 0.9669 | 5.8092 | 7.9161 |
| Previous day | 0.9188 | 1.2451 | 7.2458 | 9.9777 |

Both forecasts use the same 365 complete validation profiles. The GRU has 12,939 parameters and reached its best hourly validation MAE at epoch 10. Its predictions share one past sequence. For each horizon, the model also receives the target calendar, horizon index and target-aligned reading from the previous day. That previous-day reading is available at the common forecast origin. The previous-day baseline copies the prior 24 completed readings.

The initial description omitted the target-aligned previous-day input. This was corrected when the saved training configuration was checked for the MAPE comparison.

## Four hourly starters on validation

| Model | Parameters | Best epoch | MAE | RMSE | Training seconds |
|---|---:|---:|---:|---:|---:|
| LSTM | 17,225 | 5 | 0.6711 | 0.9083 | 23.1 |
| GRU | 12,937 | 5 | 0.6652 | 0.9048 | 18.4 |
| TCN | 37,353 | 5 | 0.6563 | 0.8901 | 258.1 |
| Transformer | 25,513 | 5 | 0.6571 | 0.8896 | 60.4 |
| Previous day | | | 0.9181 | 1.2444 | |

The four runs share the same validation targets and settings. Architecture sizes differ. These CPU timings describe one machine and vary with concurrent load.

## Frozen next-hour evaluation

All four training runs captured the exact input split hashes, code and environment. Each holdout CSV passed its manifest hash check. The four models scored identical target keys and baselines within each period. Every cell shows MAE / RMSE.

| Period | Hours | Previous day | LSTM | GRU | TCN | Transformer |
|---|---:|---:|---:|---:|---:|---:|
| Test | 8,760 | 0.9052 / 1.2200 | 0.6849 / 0.9265 | 0.6790 / 0.9196 | 0.6685 / 0.9059 | 0.6720 / 0.9055 |
| Evaluation 1 | 2,208 | 0.9878 / 1.2765 | 0.7359 / 0.9490 | 0.7313 / 0.9459 | 0.7175 / 0.9348 | 0.7277 / 0.9400 |
| Evaluation 2 | 1,416 | 1.0115 / 1.3162 | 0.7714 / 1.0189 | 0.7670 / 1.0182 | 0.7609 / 1.0078 | 0.7808 / 1.0236 |
| Evaluation 3 | 2,205 | 0.9047 / 1.1672 | 0.7112 / 0.8973 | 0.7068 / 0.8884 | 0.6953 / 0.8777 | 0.6917 / 0.8761 |

All four models improved MAE and RMSE over the previous-day baseline in all four periods. The TCN had the lowest MAE in three periods. The Transformer had the lowest MAE in Evaluation 3 and the lowest test RMSE. Evaluation 2 had the highest MAE for each model. The results use a single house and seed and do not establish a universal architecture ranking.

### Large-error days and local-hour checks

Each hourly report ranks the five complete UTC target-end dates with the largest daily MAE and records the baseline error and mean signed error. The local artifacts retain dates and diagnostic results. Partial days at the edges of the source are excluded from this ranking.

| Period | Worst TCN day MAE | Previous-day MAE on that date | TCN mean signed error | Highest-error Helsinki target hour for TCN |
|---|---:|---:|---:|---:|
| Test | 1.3336 | 2.0042 | -0.5164 | 20:00 (1.3662) |
| Evaluation 1 | 1.0602 | 1.0546 | -0.3437 | 20:00 (1.1818) |
| Evaluation 2 | 1.1263 | 1.3508 | -0.3542 | 20:00 (1.5572) |
| Evaluation 3 | 1.0307 | 1.2183 | -0.3191 | 21:00 (1.1804) |

For the four TCN days above, mean consumption and mean absolute hour-to-hour change both exceeded their respective period averages. Hours in the top quarter of each period's load distribution accounted for 62–79% of absolute error on those days. The mean signed errors show underprediction on each day. This describes where errors accumulated; it does not identify the cause of the demand changes.

## Frozen 24-hour evaluation

One GRU forecast is issued at 00:00 UTC. The profile is evaluated on the next 24 hourly values and their total. Every cell shows MAE / RMSE.

For the requested four-model daily comparison, the LSTM, GRU, TCN and Transformer use the same 24-hour input history, target calendar, horizon and target-aligned previous-day reading. Each uses seed 42, up to 10 epochs, patience 3, and the checkpoint with the lowest validation hourly MAE. MAPE is reported as an additional metric and does not select the checkpoint. These settings were recorded before evaluating the new daily models. The group had already seen the earlier starter holdout results.

The daily runs used the local `audit_20261110_v1` dataset manifest with SHA-256 `ba04e35ce5e076ad87c4c079031a723f434b1582bafb5860376294b0181b3db3`. Its 24 CSV hashes match the previous no-weather build. The LSTM, GRU, TCN and Transformer selected epochs 10, 10, 8 and 7, respectively. Their parameter counts are 17,227, 12,939, 37,355 and 25,515. Each run stores its fit/validation data, code and environment hashes with its frozen artifacts.

| Validation model | Hourly MAE / RMSE / MAPE | 24-hour total MAE / RMSE / MAPE |
|---|---:|---:|
| Previous day | 0.9188 / 1.2451 / 70.25% | 7.2458 / 9.9777 / 16.37% |
| LSTM | 0.7237 / 0.9694 / 59.40% | 5.9444 / 8.0408 / 13.69% |
| GRU | 0.7246 / 0.9669 / 59.99% | 5.8092 / 7.9161 / 13.34% |
| TCN | 0.7275 / 0.9727 / 59.84% | 6.0654 / 8.2534 / 13.85% |
| Transformer | 0.7230 / 0.9699 / 57.63% | 6.2423 / 8.4131 / 14.16% |

| Period | Complete profiles | GRU hourly | Previous day hourly | GRU 24-hour total | Previous day 24-hour total |
|---|---:|---:|---:|---:|---:|
| Test | 364 | 0.7251 / 0.9670 | 0.9054 / 1.2202 | 6.0430 / 8.1539 | 7.1044 / 9.6949 |
| Evaluation 1 | 91 | 0.7723 / 0.9865 | 0.9827 / 1.2688 | 6.9064 / 8.9031 | 7.3957 / 10.2037 |
| Evaluation 2 | 58 | 0.7927 / 1.0564 | 1.0125 / 1.3186 | 7.7775 / 9.6608 | 6.9321 / 8.8825 |
| Evaluation 3 | 91 | 0.7693 / 0.9361 | 0.9089 / 1.1706 | 5.3511 / 6.3462 | 5.3733 / 6.6929 |

The daily GRU improved hourly MAE in all four periods. In Evaluation 2 it had higher 24-hour total error than the previous-day baseline. Checkpoint selection used validation hourly MAE. The daily total is a separate outcome.

## MAPE and baseline improvement

This comparison pools the test period and all three evaluation periods, weighting every scored target equally. All models in each table use the same targets as their baseline. The hourly models are the existing frozen starters. The four daily models were trained and frozen with the previously selected fixed-origin inputs before their new scores were calculated. The group had already seen earlier holdout results.

MAPE is 100 times the mean of the absolute error divided by the absolute actual. Zero actuals are excluded from MAPE and counted. There are no zero actuals in these pooled comparisons. For daily totals, the 24 actuals and predictions are summed before calculating the percentage error. Small actual values can have a large influence on hourly MAPE.

An error reduction is 100 × (baseline error − model error) / baseline error. Positive values mean lower error. Negative values mean higher error. A MAPE reduction is relative to baseline MAPE and is not a percentage-point difference.

### Next-hour forecasts

Pooled over 14,589 hourly targets.

| Model | MAE | RMSE | MAPE | MAE reduction | MAPE reduction |
|---|---:|---:|---:|---:|---:|
| Previous-day baseline | 0.9279 | 1.2306 | 62.71% | — | — |
| LSTM | 0.7050 | 0.9350 | 49.72% | 24.03% | 20.71% |
| GRU | 0.6996 | 0.9290 | 48.00% | 24.60% | 23.46% |
| TCN | 0.6889 | 0.9165 | 47.85% | 25.76% | 23.69% |
| Transformer | 0.6940 | 0.9186 | 48.58% | 25.22% | 22.53% |

The period table shows relative error reduction for MAE / MAPE.

| Period | Baseline MAE / MAPE | LSTM | GRU | TCN | Transformer |
|---|---:|---:|---:|---:|---:|
| Test | 0.9052 / 67.45% | +24.34% / +21.52% | +24.99% / +24.44% | +26.15% / +24.58% | +25.76% / +23.61% |
| Evaluation 1 | 0.9878 / 52.63% | +25.50% / +20.86% | +25.96% / +23.16% | +27.36% / +24.89% | +26.33% / +22.66% |
| Evaluation 2 | 1.0115 / 36.00% | +23.73% / +21.97% | +24.17% / +24.17% | +24.77% / +25.32% | +22.81% / +21.19% |
| Evaluation 3 | 0.9047 / 71.10% | +21.39% / +17.10% | +21.88% / +19.74% | +23.15% / +18.93% | +23.55% / +18.77% |

### 24-hour forecasts, individual hourly targets

Pooled over 14,496 hourly targets within 604 profiles.

| Model | MAE | RMSE | MAPE | MAE reduction | MAPE reduction |
|---|---:|---:|---:|---:|---:|
| Previous-day baseline | 0.9278 | 1.2301 | 62.65% | — | — |
| LSTM | 0.7459 | 0.9785 | 54.39% | 19.61% | 13.19% |
| GRU | 0.7454 | 0.9744 | 54.97% | 19.67% | 12.27% |
| TCN | 0.7510 | 0.9859 | 54.65% | 19.06% | 12.78% |
| Transformer | 0.7428 | 0.9732 | 53.04% | 19.94% | 15.34% |

The period table shows relative error reduction for MAE / MAPE.

| Period | Baseline MAE / MAPE | LSTM | GRU | TCN | Transformer |
|---|---:|---:|---:|---:|---:|
| Test | 0.9054 / 67.36% | +19.96% / +13.98% | +19.91% / +12.90% | +19.36% / +13.03% | +20.18% / +16.74% |
| Evaluation 1 | 0.9827 / 52.52% | +20.91% / +17.01% | +21.41% / +16.58% | +21.06% / +17.81% | +21.42% / +17.75% |
| Evaluation 2 | 1.0125 / 35.72% | +21.44% / +23.80% | +21.71% / +23.21% | +19.90% / +24.22% | +21.77% / +22.49% |
| Evaluation 3 | 0.9089 / 71.13% | +15.53% / +3.97% | +15.36% / +3.19% | +15.09% / +4.47% | +16.09% / +5.97% |

### 24-hour forecasts, summed 24-hour totals

Pooled over 604 complete profiles.

| Model | MAE | RMSE | MAPE | MAE reduction | MAPE reduction |
|---|---:|---:|---:|---:|---:|
| Previous-day baseline | 6.8709 | 9.3089 | 14.69% | — | — |
| LSTM | 6.4480 | 8.4760 | 13.43% | 6.16% | 8.59% |
| GRU | 6.2354 | 8.1878 | 13.11% | 9.25% | 10.72% |
| TCN | 6.7458 | 8.8843 | 13.85% | 1.82% | 5.75% |
| Transformer | 6.4252 | 8.3946 | 13.19% | 6.49% | 10.18% |

The period table shows relative error reduction for MAE / MAPE.

| Period | Baseline MAE / MAPE | LSTM | GRU | TCN | Transformer |
|---|---:|---:|---:|---:|---:|
| Test | 7.1044 / 16.68% | +12.52% / +13.57% | +14.94% / +15.38% | +9.06% / +10.70% | +10.65% / +13.53% |
| Evaluation 1 | 7.3957 / 12.23% | -0.31% / +2.78% | +6.62% / +8.78% | -3.25% / +2.91% | +4.57% / +6.91% |
| Evaluation 2 | 6.9321 / 8.91% | -16.67% / -10.02% | -12.20% / -6.70% | -33.14% / -24.72% | -17.81% / -12.39% |
| Evaluation 3 | 5.3733 / 12.86% | +0.17% / -3.53% | +0.41% / -3.87% | -0.76% / -3.80% | +7.09% / +5.89% |

In Evaluation 2, all four daily-total models had higher MAE than the previous-day baseline. The pooled daily-total MAE reduction is positive for each model. The comparison uses one seed and one house.

The detailed local results and reproducible commands are described in the [model guide](models/README.md).

## Remaining group work

The four starters and their local HTML exports are available. Each student still needs to adapt and explain their own implementation, document AI use and reproduce the shared data and comparison. The group can then complete its final report. The split timezone and consumption unit remain provisional or unspecified. Historical weather availability and suitable same-origin forecast vintages remain unverified.
