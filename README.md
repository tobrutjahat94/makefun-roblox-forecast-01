# Roblox Up-and-Coming forecast — make.fun entry

Entry for `roblox-up-and-coming-forecast-01`: predict the Up-and-Coming
chart's top 10 at every hour of the 72 hours after entries close.

## Method

`main.py` (Python 3.13, stdlib only, no network, deterministic) models each
game in the current top 60 as a Gaussian rank process over the window:

- **Rank drift + diffusion.** A game's rank at a future hour `t` is
  `Normal(mu, sigma)` with `mu = r0 + drift*t` and `sigma = 2.5*sqrt(t)`.
  The drift blends the game's own recent rank velocity (linear fit over
  the last ~30h, weight 0.5) with a rank-bucket drift plus a tenure
  adjustment: new games climb, old games fall faster (clipped to
  +/-0.35 ranks/hour).
- **Time-averaged probabilities.** `P(game at place k)` is the normal mass
  on `[k-0.5, k+0.5]`, averaged uniformly over `t = 0, 3, ..., 72` hours.
  Per place the probabilities are normalised and the top 20 games are
  named; the rest is the unnamed share (typically 0.15–0.28).
- **`playing` counts are not used**: measured correlation with rank change
  is ~0 (they are diurnal, not a rank signal). Rank momentum is the real
  signal (corr ~0.28).
- `top10` is the argmax game per place (tie-break only).

Parameters were fitted on the 34-snapshot archive (2026-10-30 to
2026-11-07) and validated by backtesting.

## Backtest

21 cutoffs, scored on the reads within 72h with the exact grader log-score
rule (mean over reads of the sum over 10 places of `ln(p)` given to the
game actually at that place, floored at `ln(1e-4)`):

| forecaster | mean log score |
|---|---|
| house persistence baseline | -48.53 |
| this entry (V3) | **-37.36** |

V3 beats the baseline by **+11.2 log points** and wins 21/21 cutoffs.
The entry program itself was verified against the backtest reference
implementation: identical probabilities (max diff 0.00000) and identical
top10 on 4 spot-check cutoffs, mean -36.79 vs baseline -46.18.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads `history.ndjson` + `task.json` from the input dir, writes
`forecast.json` to the output dir. The window (horizon) is read from
`task.json`, not hardcoded. Two runs on the same history produce byte-identical
output.

## License

MIT. See `LICENSE`.
