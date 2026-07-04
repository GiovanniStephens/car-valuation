# Product Roadmap — from valuation engine to deal-finder website

> **Status:** exploratory / thinking-out-loud. This is a vision doc, not a commitment.
> It sketches how the current codebase could grow from a set of local scripts into a
> hosted product. Nothing here is decided; it's here to argue with.

## The one-line idea

The valuation model is the **engine**, not the product. The product is a
**deal-detection and alert service for New Zealand's second-hand car market**:

> *"This specific listing is priced below what it's actually worth — here's the estimate,
> the confidence, and the link. Move now."*

Plain valuation ("what's my car worth") is commodity — OneRoof / TradeMe's own tools
give estimates already. The value is in **finding underpriced listings fast and telling
someone before the deal is gone.** `find_undervalued.py` is already a prototype of exactly
this; the product is the layer that runs it continuously and puts it in front of buyers.

---

## Where we are today (honest snapshot)

What already exists in this repo is most of a real valuation engine:

| Capability | Where | Product relevance |
|---|---|---|
| **Official TradeMe API ingest** (OAuth, not scraping) | `utils.fetch_trademe_data`, `requests-oauthlib` | The legitimate, durable data path — de-risks the whole thing vs. HTML scraping |
| **Per-model AutoGluon valuation** | `training.py`, `inference.py` | Core price estimate |
| **Confidence intervals** (quantile model, P10–P90) | `inference.py` | Lets us score *deal quality* honestly, not just point estimates |
| **General-model blending** for thin data | `train_general_model.py`, blend rule in `inference.py` | Coverage across long-tail makes/models |
| **Undervalued batch search** w/ min/max discount + region compare | `find_undervalued.py`, `undervalued_search.yml` | This *is* the product's core loop, in prototype form |
| **Interactive dashboard** | `dashboard.py` (Streamlit) | v0 of the UI |

**What's missing to be a product** (all of it is plumbing, not research):
continuous/scheduled ingest, a database of listings over time, hosted models,
user accounts, saved searches, and an alerting channel. None of that is hard; it's
the difference between "a script I run" and "a service that runs for other people."

---

## Who it's for (and the wedge)

| Segment | Willingness to pay | Notes |
|---|---|---|
| 🎯 **Car flippers & small dealers** | **High** | Clear ROI — one good flip pays for a year. Recurring need for underpriced stock. **Start here.** |
| Private buyers (buying one car) | Low–med | Big audience, churns after buying. Great free top-of-funnel, weak core. |
| Trade-in / valuation seekers | Low | Commodity lookup. Free tier / SEO magnet. |

**Wedge: flippers and small dealers, cars only.** Cars transact often, so the model
gets fast feedback and the deal flow is continuous. Property (see the sister repos
`house-price-scraper` / `house-price-tracker`) is the natural **second asset class**
under the same "deal finder" brand — later, not now.

---

## Evolution phases

Each phase is independently useful — you get value even if you stop at any one.

### Phase 0 — Today: local research tool
Scripts + local models + Streamlit, run by hand. Config in YAML, creds in `keyring`.
**Good enough to validate the core question:** *do the flagged "undervalued" cars turn
out to be genuine deals?* (See "Validate first" below — do this before building anything.)

### Phase 1 — Personal hosted tool (single user: you)
Get it running **unattended** for your own use.
- Move `find_undervalued.py` onto a **schedule** (cron / GitHub Action, mirroring the
  `house-price-tracker` daily-job pattern).
- Persist every listing + estimate to a small **database** (SQLite → Postgres/Supabase —
  `NZ_Housing_Market_Dash` already uses Supabase, so there's a pattern to copy).
- Push results to **you** via email / Telegram / Discord.
- **Outcome:** a daily "here are today's underpriced cars" digest. Proves the pipeline
  runs without you, and starts accumulating the dataset (the eventual moat).

### Phase 2 — Freemium public website
Turn it outward.
- **Free tier:** single valuation lookup ("what's this car / my car worth"), with the
  confidence interval. This is the SEO magnet and top-of-funnel.
- **Paid tier:** saved watchlists (make / model / budget / region) → **real-time alerts**
  + a ranked live "deals" feed. Each deal shows estimate, confidence, discount %, comps,
  and the listing link.
- Reuse `dashboard.py` as the v0 UI; add auth + billing.
- **This is the first version that can make money.**

### Phase 3 — Multi-asset + data product
- Add **property** as a second asset class (plug in `house-price-scraper`'s AVM).
- Sell **market insights** off the accumulated data: days-on-market, ask-vs-sold,
  depreciation curves, regional arbitrage — to dealers, insurers, and researchers.
- Optional: a **valuation API** for B2B (brokers, insurers, other apps).

---

## Architecture evolution

The jump from Phase 0 to Phase 2 is mostly moving state out of the filesystem and
wrapping the engine in a service.

| Concern | Today | Needs to become |
|---|---|---|
| **Ingest** | manual `--fetch` | scheduled worker (cron/queue) pulling new listings continuously |
| **Storage** | CSV / local files | database: listings + estimates + outcomes as time series |
| **Models** | `models/` on disk, `model_cache.py` | model store + periodic retraining job; version + track drift |
| **Valuation** | in-process function call | internal service / API endpoint |
| **Deal logic** | `find_undervalued.py` batch run | streaming scorer that runs on each new listing |
| **UI** | Streamlit run locally | hosted web app (keep Streamlit for v0; Next.js later — see `JobMarketIQ` for a full-stack pattern) |
| **Auth / billing** | none | accounts + Stripe (or similar) for the paid tier |
| **Alerts** | none | email / SMS / push / Telegram fan-out per saved search |
| **Secrets** | `keyring` (local) | proper secret manager; one app-level TradeMe credential |

Nothing here is novel engineering. The critical-path items are **scheduled ingest → DB →
alerting**; auth/billing only matter once other people use it (Phase 2).

---

## Monetisation

- **Freemium SaaS.** Free valuation lookup; paid subscription for alerts + deal feed.
- Indicative tiers (validate against real willingness to pay):
  - *Buyer* — a few $/mo: N saved searches, daily alerts.
  - *Pro / dealer* — a few tens of $/mo: unlimited searches, faster refresh, region arbitrage, API.
- Later: **data / insights** sold to dealers & insurers; **per-lookup API** for B2B.

Be honest about scale: NZ is ~5M people and flippers are a niche within that. Realistic
target is a **profitable lifestyle SaaS**, not venture-scale — unless it later crosses to
Australia (10× the market, but incumbents like RedBook/CarsGuide are entrenched).

---

## The moat

Every listing observed over time — ask price, spec, region, days-on-market, whether it
sold, and how the estimate moved — becomes a **proprietary NZ car-market dataset nobody
else has.** It compounds while idle: the model sharpens, and eventually the *data itself*
is sellable. Phase 1 is worth doing partly just to **start accumulating this now.**

---

## Validate first (before building the business)

Two questions decide whether any of this is real. Answer them cheaply, first.

1. **Does the deal-detection actually work?**
   Take `find_undervalued.py`'s flagged cars and check: are they genuine deals, or model
   error / hidden write-offs / wrong-variant mismatches? Measure precision on a sample.
   If a large share of "40% underpriced" hits are noise or lemons, the alerts are
   worthless and everything above is moot. **This is the make-or-break test.**

2. **Do TradeMe's API terms permit commercial use / redistribution of listing data?**
   We use the official API (good), but the *terms* may restrict commercial products or
   redistributing listing data to end users. Read them; if needed, ask TradeMe about a
   commercial tier or partnership. The entire product's legality/durability rests on this.

Also worth a look: TradeMe already runs its own valuation feature — know how ours differs
(multi-model blending, honest confidence intervals, **deal-finding** rather than single
estimates) and where it genuinely beats theirs.

---

## Immediate next steps (concrete, small)

- [ ] **Precision check** on `find_undervalued.py` output — manually grade ~30–50 flagged deals. (Answers Q1.)
- [ ] **Read the TradeMe API terms** for commercial/redistribution limits. (Answers Q2.)
- [ ] **Persist listings to a DB** instead of CSVs — start the time series now (cheapest possible moat-builder).
- [ ] **Schedule `find_undervalued.py`** to run daily and email you the results (Phase 1 in a weekend).
- [ ] Only then: decide whether Phase 2 (public freemium site) is worth building.

---

## Open questions

- Cars-first is the call — but is there a sharper wedge *within* cars (e.g. one segment:
  Japanese imports, or a price band, or a region) to dominate before broadening?
- Alert-product vs. data-product: which do NZ dealers actually pay for? (Ask a few.)
- How fast is "fast enough" for alerts — do good deals really vanish in hours, or days?
- Is the honest ceiling a lifestyle SaaS, and is that the goal — or is this a portfolio /
  credibility piece, or a springboard to the AU market?
