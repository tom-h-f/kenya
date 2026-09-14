
## Built 2026-09-14

`kenya_monitor.control`, `monitor control`, `config/control_frame.yaml`,
`control_runs/`, and a fourth value in `kma.db.SCOPES`.

Decisions the implementation settled, with the argument for each:

- **The frame ships as the Kenyan-politics one** - institutions and offices
  (IEBC, Bunge, Parliament, Senate, National Assembly, Governor, Senator,
  county government, by-election, voter registration) conjoined with a Kenya
  anchor group. Not campaign language: a frame built from slogans
  over-represents slogan talk, which is the bias `targets.yaml` has and this
  arm exists to avoid. Swap it by editing the file, but a change starts a NEW
  frame and rates either side of it are not comparable.
- **Windows are aligned to the epoch, not to `now`.** Two passes minutes apart
  must see the same grid, or "already sampled" is unanswerable and each pass
  draws from a different population.
- **`SEARCH_PRODUCT` is already `Latest`.** The census claim needed
  reverse-chronological results rather than relevance ranking, and the
  collector has been set that way since long before this - so the design's
  central assumption was already satisfied rather than newly assumed.
- **Empty windows are recorded.** The denominator is sampled windows, so a
  window with no frame posts is an observation; dropping it would bias every
  rate upward.
- **Truncated windows are recorded, kept, and flagged.** A window at the cap is
  a ranked sample of an unknown larger set and the census claim does not cover
  it. Dropping them instead would bias the sample toward quiet windows, which
  is worse than knowing.
- **`all` still means what it meant.** The control arm is reachable only by
  naming the `control` scope. Adding a partition must not silently change what
  an existing number means.
- **In the cycle from the start**, unlike the depth arms. X search reaches 14
  days, so a one-off pass samples the last fortnight and nothing else; coverage
  of the collection period accumulates only by running continuously.

## The one number still to measure

Whether a 15-minute window of this frame fits under the 100-post cap. If it
does, each sampled window is a census and the arm works as designed. If it does
not, `truncated` will say so on the first pass and the window narrows until it
stops. That measurement needs the account pool, so it needs a deploy - it
cannot be taken from a laptop.
