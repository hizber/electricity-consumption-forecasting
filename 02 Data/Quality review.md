# Fitting-data quality review and evaluation coverage

**Status:** Initial review on 7 November 2026, with observation and coverage counts audited on 10 November 2026. The inspection below uses fitting-period source readings from 1 January 2016 00:00 to 1 October 2019 00:00 UTC, end exclusive. The verified source SHA-256 is `9e8334d93bc3f9f1391216f68f52eedb5c9ebd3c2a5856e97db0177cf76f70a0`. Later-period labels were not used to select features or decide how to treat anomalies. Weather observations in this review are descriptive context, with no assumed forecast-time availability.

## Review of high consumption readings

The EDA uses the fitting-period distribution with Q1 = 0.69 and Q3 = 2.64. The upper 1.5 × IQR fence is 5.565 in the source's unspecified consumption units. It flags 525 of 32,856 fitting-period readings, about 1.60%. This is a screening rule. It does not establish that a meter reading is wrong.

Four of the largest distinct peaks were inspected using five-hour windows around each one. The windows show sustained high readings around some peaks and sharper changes around others, but do not establish appliance activity, causes or measurement accuracy. All four peaks occur in winter, and 370 of the 525 flags fall in January or February in Helsinki local time. Seasonal variation is one reason that a single global fence can label legitimate high-demand hours. The 2016 weather provenance remains unresolved because the assignment describes weather coverage starting in 2017. The weather station is not a verified house location.

I also compared the two largest rises and the two largest falls against five-hour source windows. The source values match the flagged changes. The next-hour change reverses direction in three of the four cases. Exact timestamps and readings are in the ignored local file `02 Data/Electric-consumption/quality-review-examples.md`.

Retain the observations for the first experiments. The review found no evidence that supports correcting the sampled records. Regenerating the fitting-only EDA produced 4,259 feature-event flags on 3,411 distinct target-end timestamps. The one-hour consumption change contributed 991 events.

Expanding each event to the source readings that produced it gives 4,448 distinct variable-and-timestamp observations. These comprise 2,559 consumption readings, 282 temperature readings, 664 humidity readings and 943 wind readings. The consumption count includes every reading in flagged 24-hour rolling windows and both readings in flagged changes. It does not mean all those individual readings are anomalous. Different features can flag the same target timestamp, and one source reading can contribute to multiple events.

## Unit and time coverage

The supplied column dictionary does not state a consumption unit. Report results in **source units** until an authoritative source confirms a unit. The source ends at **31 May 2022 20:00 UTC**, before the end of that UTC calendar day. Assignment split dates are interpreted as UTC provisionally and should still be confirmed.

The following counts and endpoints come from the local independently verified no-weather rebuild. Hourly counts are forecast targets. Daily counts are the number of complete 24-hour profiles issued at 00:00 UTC; each profile contains 24 target rows and must fit inside one split.

| Period | Hourly targets | Hourly targets absent | Last hourly target end, UTC | Complete daily profiles | Daily origins excluded | Last daily target end, UTC |
|---|---:|---:|---|---:|---:|---|
| Test | 8,760 | 0 | 2021-09-30 23:00 | 364 | 1 | 2021-09-30 00:00 |
| Evaluation 1 | 2,208 | 0 | 2021-12-31 23:00 | 91 | 1 | 2021-12-31 00:00 |
| Evaluation 2 | 1,416 | 0 | 2022-02-28 23:00 | 58 | 1 | 2022-02-28 00:00 |
| Evaluation 3 | 2,205 | 3 | 2022-05-31 20:00 | 91 | 1 | 2022-05-31 00:00 |

The hourly exclusions compare scored target timestamps with every hour in each provisional UTC period. Daily exclusions compare scored profiles with all 00:00 UTC origins dated within each period. The final origin in the test period and in evaluations 1 and 2 is excluded because its 24th target crosses the split boundary. The final Evaluation 3 origin has an incomplete profile because the source ends at 20:00 UTC. Do not fill the missing hours. Recalculate all counts if the timezone or schedule changes.

The separately verified rebuild's output hashes matched the earlier verified dataset. A direct workspace rebuild previously produced empty hourly evaluation files; the pipeline has since been updated to mitigate the write problem, while the underlying cause remains unconfirmed. Verify a fresh output and its hashes before training or reporting.
