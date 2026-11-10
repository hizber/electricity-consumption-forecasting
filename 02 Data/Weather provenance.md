# Weather provenance and public sources

**Status:** Research checked on 7 November 2026. Historical publication times, revision vintages and the origin of the 2016 course-file weather values remain unverified. This note documents an investigation; it does not establish that the experimental weather features were available at every historical forecast origin.

## Course data

- The course CSV already contains hourly weather observations including air temperature, humidity and wind. The accompanying dictionary labels `MTime` as UTC and describes air temperature as an hourly average. The supplied data-generation script names FMI station `101339`, Jyväskylä Airport, and queries hourly observations. This shows the intended source in that script; it does not independently prove which script version produced every row of the supplied CSV. The recorded coordinates identify the **station**, not a verified house address. See [[Data inventory]] and [[Initial exploration]].
- The assignment PDF describes hourly weather as starting on 1 January 2017. The inspected course CSV contains air temperature in 8,782 of 8,784 UTC-dated hours in 2016. The reason for the difference has not been established. Focused FMI WFS queries on 7 November 2026 for station `101339`, using `fmi::observations::weather::hourly::multipointcoverage` for 00:00–05:00 UTC on 1 January in 2016 and 2017, each returned zero features. That result does not establish whether the station had observations or whether the query was suitable. The supplied values have not been independently matched to the archive. The course CSV hash is `9e8334d93bc3f9f1391216f68f52eedb5c9ebd3c2a5856e97db0177cf76f70a0`.

## Availability at a forecast origin

FMI describes hourly observations as based on the previous hour. Its current past-weather display usually updates every ten minutes, but FMI notes that update schedules are not guaranteed. That current display cadence does not establish the release time or revision history of each 2016–2022 historical observation. No checked source establishes that the experimental one-hour publication delay holds for the full course period. [FMI observation guidance](https://en.ilmatieteenlaitos.fi/guidance-to-observations), [FMI weather service updates](https://en.ilmatieteenlaitos.fi/weather-services-on-the-internet).

The existing no-weather datasets remain the primary comparison. A lagged-weather experiment may use a documented delay as a sensitivity assumption, with results labelled conditional until observation availability and revisions are verified. Observed target-hour weather is not known at the forecast origin. For a once-daily forecast, all 24 hours must use information available by the common origin.

## Candidate public sources

| Source | What is documented | Checks before use |
|---|---|---|
| [FMI observations](https://en.ilmatieteenlaitos.fi/download-observations-questions) | Observation downloads use UTC and the FMI open-data licence, CC BY 4.0. FMI offers hourly historical observations where a station and variable have coverage. | Verify station, coverage, units, interval meaning, source versions, release and revision times. Compare downloaded records with the course CSV before merging. Do not treat a current archive retrieval as evidence of historical publication time. |
| [Open-Meteo previous runs](https://open-meteo.com/en/docs/previous-runs-api) | Fixed lead-time series are provided at one to seven days before valid time. Most listed models begin in 2024; GFS 2 m temperature starts March 2021 and JMA GSM/MSM in 2018. | The global JMA GSM may cover Finland, whereas MSM is regional to Japan/Korea. Check actual location, variable and time coverage. GSM values are interpolated from six-hourly to hourly. Data prior to 2018 are not covered by this documented option. Verify availability latency and original vintage before use. [JMA model details](https://open-meteo.com/en/docs/jma-api). |
| [Open-Meteo single runs](https://open-meteo.com/en/docs/single-runs-api) | Full model runs are selected by initialization time. The service lists ECMWF IFS from March 2024 and other models from April 2026. | These documented dates are later than the 2016–2022 project data, so this archive alone does not provide the required historical runs. Initialization time also needs an actual publication-delay check. |
| [Open-Meteo historical forecast](https://open-meteo.com/en/docs/historical-forecast-api) | A continuous series stitches the first few hours of successive model runs. | It cannot automatically represent all 24 hours of one forecast that was available at a single earlier origin. Reconstruct using verifiable vintages instead. |

Open-Meteo states that its API data are available under CC BY 4.0 and lists underlying providers and their licences. Check terms and attribution for the actual model and usage before downloading or sharing. [Open-Meteo licence](https://open-meteo.com/en/licence).

## Remaining work

1. Obtain reliable evidence for the course weather's 2016 provenance and check archive values without exposing household readings.
2. Establish available historical publication and revision information. Where it cannot be established, state the delay assumption and assess sensitivity using longer lags.
3. For any external source, record query parameters, source, licence, station/grid location, coverage, timestamps, units, missingness, retrieval date, revision or forecast vintage, and a content hash. Check alignment and differences before a reproducible join.
4. The [conditional validation experiment](../04%20Experiments/Starter%20results.md) compares past weather and no weather on the same target hours, with fitting-only imputation. Its results use a one-hour delay assumption. The primary held-out comparison excludes weather.
