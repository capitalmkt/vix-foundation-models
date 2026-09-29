# Data

The CSV files are not included in this repository. Live downloads are
disabled by design in every script (see the module docstrings), so you need
to place two approved CSV files here yourself before running anything.

## VIX.csv (required for every script)

Daily closing values of the Cboe Volatility Index.

Accepted schemas, either one:

| Date column | Value column |
|---|---|
| `Date` | `Close` |
| `DATE` | `VIXCLS` |


Sources:
- Cboe official historical data page: https://www.cboe.com/en/tradable-products/vix/vix-historical-data
- FRED (same series, ticker VIXCLS): https://fred.stlouisfed.org/series/VIXCLS

## VVIX.csv (required only for the scripts in src/multivariate)

Daily closing values of the Cboe VIX of VIX Index, used as the second series
in the VIX+VVIX information set.

Accepted schemas, either one:

| Date column | Value column |
|---|---|
| `Date` | `Close` |
| `DATE` | `VVIX` |


Source:
- Cboe official historical data page (see "Cboe VIX of VIX Index (VVIX)"): https://www.cboe.com/en/tradable-products/vix/vix-historical-data
- Investing.com CBOE Vix Volatility (VVIX): https://www.investing.com/indices/cboe-vix-volatility

## Where to put the files

Place both files directly in this `data/` folder:

```
data/
├── VIX.csv
└── VVIX.csv
```

Each script also accepts a few alternative filenames; see `LOCAL_VIX_PATHS`
and `LOCAL_VVIX_PATHS` near the top of each script for the exact list.
