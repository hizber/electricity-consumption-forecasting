# Data inventory

The prioritized [[Data preparation checklist]] tracks the remaining work before and after the first model runs.

**Status:** The live Moodle electricity dataset was copied into the `02 Data/Electric-consumption` folder on 6 November 2026 and its CSV, dictionary and script match the historical course copy by SHA-256. The details below come from [[Group Work assignment-2026.pdf]], page 2; observed differences are documented in [[Initial exploration]].

- Moodle's Datasets tab is said to contain `Electricity_consumption.zip`, with `Dataset_mansion.csv` and `Dataset_mansion.txt` (column explanations).
- Hourly consumption is described as covering 31 December 2015 22:00 to 31 May 2022 20:00. The example says a measurement stamped 22:00 represents consumption between 21:00 and 22:00.
- Hourly weather is described as covering 1 January 2017 through 31 May 2022. Thus the stated weather coverage starts after the stated training period begins.
- House: electric boiler and water pipes inside the floor; apartment 138 m², storage 10 m², semi-heated garage 30 m². Household: two adults and three children.
- The evaluation 3 calendar range runs to 31 May 2022, but the PDF says consumption ends at 20:00 on that date. Check actual timestamps instead of assuming a complete final day.

## Inspection record
See [[Initial exploration]] and [[summary.json]]. The column dictionary states UTC; consumption unit is not specified. The CSV, dictionary and script hashes are `9e8334d93bc3f9f1391216f68f52eedb5c9ebd3c2a5856e97db0177cf76f70a0`, `1006f086db24a22d6be964dcb6a1a218fec6a076a8a6bce546817cb4e6181135`, and `142746ad7d0c77af8e21c6e0b566bd5bc627649309932cff379a44026a25f4dc`, respectively.
