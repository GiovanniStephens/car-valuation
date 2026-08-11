# PRD — Car deal-finder v1

> **Status:** decisions, not exploration. This document governs what gets built.
> For market thesis, wedge, monetisation and long-term phases, see [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) — that doc argues, this one decides. Where they disagree, this one is current.
>
> Owner: Giovanni · Last revised: 2026-08-11

## What this is

NZ second-hand car listings are routinely mispriced against their own comparables. The valuation engine in this repo can already spot that, but only when run by hand — so the signal is worthless in practice, because a good deal is gone before anyone runs the script.

**v1 is a personal, single-user tool** that watches TradeMe unattended, scores every new listing against the trained models, and alerts Giovanni when something is genuinely underpriced — which he uses to buy and sell cars himself. It is not a product anyone else can log into. That's deliberate: the roadmap's make-or-break question ("are the flagged deals real?") has to be answered before it's worth building for anyone else, and answering it needs exactly this system.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| v1 scope | Personal single-user tool | Validates deal precision before building for dealers |
| Automated source | TradeMe API only | Official OAuth API; durable and not a ToS gamble |
| Facebook | Manual paste only | No scraping, by policy — the extractor already handles pasted text |
| Negotiation | Draft-for-approval | Giovanni sends from his own account; system never sends |
| Frontend | Evolve the existing Streamlit app | Next.js is a rewrite that buys nothing while there's one user |

## What v1 must do

Priorities: **P0** = v1 doesn't exist without it · **P1** = in v1, tolerable slip · **P2** = deferred.

Already built and not re-specified here: TradeMe OAuth ingest, AutoGluon per-model valuation, quantile P10–P90 intervals, tail models, the Streamlit dashboard, and Groq extraction of pasted listing text.

### Ingest — P0
- Runs on a schedule with no human action, covering the configured make/model watchlist. *Accept:* runs unattended for 7 consecutive days with no manual intervention.
- Captures the fields TradeMe already returns but the code currently discards — photos, dealer-vs-private, listing start date, and price changes. *Accept:* a stored listing has a photo URL, a dealer flag, and a first-seen date.
- Fails loudly. A failed run notifies rather than leaving a silent gap. *Accept:* a forced API failure produces a notification.
- Target cadence: new listings observable within 30 minutes. The API is not the constraint here — at 12 make/model pairs, 15-minute polling uses roughly 5% of the 1000 calls/hour limit. Local retraining compute is the real ceiling, at ~10 min per pair.

### Store — P0
- One row per *observation*, not per listing, so price changes, days-on-market, relists and delisting dates are all recoverable. This is the moat the roadmap describes, stated as a requirement. *Accept:* a listing whose price drops yields two observations and a recoverable price history.
- Every stored estimate is stamped with model version and training cutoff, so scoring can be verified as genuinely out-of-sample. *Accept:* any historical estimate can be traced to the model that produced it.
- Results survive restart. Today's dashboard loses them. *Accept:* deals found yesterday are still listed after a reboot.

### Score — P0
This is where v1 differs most from what exists, and the reasoning is empirical rather than aesthetic. Measured against the 833-listing RAV4 snapshot on 2026-08-11:

- **The trigger is the confidence interval, not the point estimate.** `find_undervalued.py:154` currently calls `predictor.predict()` and throws away the intervals that `inference.py:106` already returns. The configured 40–80% discount band matches **2 listings out of 769**; scoring against the quantile model flags **50 below P10**. The band isn't mistuned, it's measuring the wrong thing. *Accept:* the trigger is "asking price below its predicted P10".
- **Every non-auction listing gets scored.** `feature_engineering.py:184` runs IsolationForest outlier removal whenever a `price` column is present — which on the deal-finding path is always — deleting **41 of 810** listings before scoring, and the deleted rows are precisely the cheap-for-spec ones. The deal-finder is discarding its own candidates. *Accept:* the scored count equals the non-auction listing count.
- **Rank on dollars, not percent.** `P25 − asking price` is the headline: how much is on the table even if the estimate is pessimistically wrong. Interval width varies enormously between listings, so a percentage isn't comparable between an $8k and a $40k car. *Accept:* the deal list sorts by dollar margin.
- **Compare within segment.** Dealer listings median $38,950 against private $17,998 on this snapshot; scoring a private car against a mostly-dealer comparison pool manufactures a deal out of the ordinary private-sale discount. *Accept:* a private listing is scored against a private-comparable estimate.
- A confidence tier — from training-set size, interval width, the tail-extrapolation flag, and whether the car's year and odometer fall inside the training data's range — gates what is allowed to alert. Low-confidence candidates go to the dashboard, never to the phone. *Accept:* a listing outside the training range does not alert.
- Suspiciously low listings (far below P10) route to "verify manually" rather than ranking as the best deal of the day. *Accept:* the cheapest outlier is flagged, not celebrated.
- Auction listings are excluded on `IsClassified`, not `BidCount` — only 6 of 833 rows carry a bid count, so the current filter barely filters. For auctions the displayed price is a current bid, not an ask, so scoring it as an ask is a category error. *Accept:* no auction appears in the deal list.
- Regional arbitrage is retained as a scoring dimension. **P1.**

### Alert — P0
- One channel for v1. Delivers within 5 minutes of scoring.
- The message carries enough to triage in ten seconds: photo, asking price, the estimate **as an interval**, dollar margin, region, days listed, dealer-or-private, and the link.
- A listing alerts once. It re-alerts only if its price drops materially or it relists. *Accept:* re-running the scorer sends nothing new.
- Marking a deal dismissed or interesting writes back to the store — this is what populates the grading data below, so it can't be a read-only notification. **P1.**

### Ledger — P1
- Records what actually happened: bought at, costs (WOF, repairs, grooming), sold at, days held, realised margin against the estimate at purchase.
- Do not cut this. It is simultaneously the P&L for Giovanni's own buying and selling, and the only unimpeachable source of truth for whether the model is right.

### Facebook — P1
- Paste a listing, confirm or correct the extracted fields, get a valuation with intervals, optionally save it as a tracked candidate.
- Explicit non-requirements: no scraping, no automated fetching, no bulk import. One listing at a time, human-initiated.

### Negotiation — P2
- Drafts an opening message grounded in the estimate and real comparable listings; Giovanni edits and sends it himself.
- Never auto-sends. No fabricated facts, no invented competing offers, no manufactured urgency. Every price claim traceable to a real listing.

## How we'll know it works

The roadmap names deal precision as the make-or-break question but doesn't say how to answer it. Here is the protocol.

**Grade the next 40 flagged candidates** within 48 hours of each alert, against a fixed, mutually-exclusive rubric: *genuine deal* / *variant or spec mismatch* / *disclosed defect* (damage, no WOF, deregistered) / *data error* (odometer, year, price typo) / *not comparable*.

Report precision **with the failure-mode breakdown**, because which failure dominates dictates the fix — model error means retrain or add features, disclosed defects mean better suppression rules, data errors mean validation. A bare precision number tells you nothing actionable.

**Pre-register the decision thresholds before running the test**, so the result can't be rationalised afterward:

- **≥40% genuine** — the alert thesis holds; Phase 2 becomes worth considering.
- **15–40%** — real signal, but needs suppression work before anyone else sees it.
- **<15%** — the roadmap's "everything above is moot" branch. Reconsider whether this is an alert product at all.

Two independent checks, since self-grading is biased: whether flagged cars disappear faster than the market median for their make/model (a proxy for selling), and — the gold standard, at low volume — realised sale price versus estimate-at-purchase from the ledger.

**Interval calibration is currently unmeasured.** Roughly 80% of actual prices should fall inside P10–P90. Since the whole scoring design now rests on P10, measuring this is a prerequisite, not a nice-to-have.

**v1 is done when:**
- [ ] The loop has run unattended for 14 consecutive days
- [ ] Interval calibration has been measured and is within tolerance
- [ ] 40 candidates graded, precision recorded with failure breakdown
- [ ] At least one car bought or sold through the tool, recorded in the ledger

## Not in v1

Each of these is decided, not merely unscheduled:

- Multi-user accounts, auth, billing — one user, so it's pure overhead
- Public website or Next.js frontend — Streamlit is sufficient for one person
- Any automated Facebook access — policy, not capability
- Property or any second asset class — roadmap Phase 3
- Public or B2B API, mobile app — no consumer
- Auto-sending any message to a seller — see Negotiation
- Accident history, odometer verification, mechanical inspection data — not obtainable from listing data
- Australia or any non-NZ market

## Risks

**TradeMe's commercial-use terms are unread.** v1 is single-user personal use, which is the lowest-risk reading and doesn't depend on the answer — but it is a hard gate on Phase 2, not on v1. Resolve by reading the terms before building anything outward-facing. Related: don't rehost listing photos or store seller names and contact details until that's settled.

**Deal precision may be noise.** This invalidates the entire thesis if true. Mitigated by the interval-based trigger and the protocol above — but genuinely unresolved until those 40 candidates are graded.

**Listing data can never reveal mechanical condition or accident history.** This is a permanent structural limit, not a bug to fix. Consequences for how the product speaks: it says *"priced below model estimate"*, never *"this is a good deal"*; every valuation appears as an interval with its comparison population named; and every alert carries a plate or VIN lookup link, which turns an unfixable epistemic hole into a concrete next action. It also caps how much of this can ever be automated.

**AI-drafted negotiation messages to private sellers.** Human approval always, sent under Giovanni's own name, no fabricated facts, no impersonation. The real ethical content is an asymmetry: dealers negotiate professionally, private sellers mostly don't. Open question — whether AI assistance is disclosed.

**Feedback-loop bias in the ledger.** He only buys what the model flags, so outcomes are conditioned on the model's own errors. Mitigate by also recording the outcome of flagged cars he passed on.

## Related

- [`PRODUCT_ROADMAP.md`](PRODUCT_ROADMAP.md) — strategy, market thesis, phases, monetisation
- Sister repos `house-price-scraper` / `house-price-tracker` — the second asset class, later
