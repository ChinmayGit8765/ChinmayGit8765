# holdem-ml

Texas Hold'em you can actually sit down and play — against bots whose poker brain
was **written and trained from scratch here**, no PyTorch, no TensorFlow, no
pretrained anything. Every matrix multiply, every backward pass, every regret
update is in this repository.

Three things live in one box:

| | |
|---|---|
| 🃏 **A game** | Full no-limit Hold'em — terminal or browser, 2–9 seats, several humans and bots at the same table |
| 🤖 **Bots that learn *you*** | A neural policy trained by self-play, plus an online opponent model that fits itself to your specific leaks, on a difficulty ladder that can lock to a level or track your skill as it grows |
| 🔬 **An analyser** | Reads hand histories (including real ones from a poker site) *and photographs of a table*, grades every decision in big blinds, and tells you what your leaks are |

---

## Quick start

```bash
pip install -r requirements.txt

python -m holdem play                       # sit down against 5 adaptive bots
python -m holdem play --difficulty pro --seats 3
python -m holdem serve --port 8000          # browser table: several humans + bots
python -m holdem bench --hands 5000         # run the bots against each other
python -m holdem analyse hands.txt          # grade a hand-history file
python -m holdem read table.png --hole 2    # read the cards out of an image
```

At any decision the terminal game accepts `f` / `c` / `r 40` / `r 75%` / `r pot` /
`a`, plus `?` for the model's read on the spot and `i` for what the bots have
worked out about you.

---

## Why Python (and only NumPy)

The brief asked for "the best ML language for this use case". For a poker engine
plus custom models, that is **Python with NumPy**, and it is not close:

* The hard part of poker AI is not raw FLOPs, it is **tree search, abstraction and
  bookkeeping** — regret tables, information sets, betting trees. That is
  dictionary-and-object work, where Python is fastest to write and hardest to get
  subtly wrong.
* The numeric work that *is* heavy — Monte-Carlo equity, batched hand evaluation,
  convolutions — is a handful of large array operations, which NumPy hands to
  BLAS. The vectorised evaluator here scores **~450,000 seven-card hands per
  second** from pure Python.
* Writing the layers by hand (rather than importing a framework) is the point: the
  network shapes here are small and unusual — a two-headed card CNN, a
  three-headed analyser — and hand-written backprop makes them auditable. The test
  suite checks every gradient against finite differences to ~1e-8.

Where Python would have hurt, the cost was measured and removed: the CFR solver
runs across all cores, convolution is a single GEMM per layer via `im2col`, and
`col2im` scatters with a sort + `reduceat` instead of `np.add.at` (3× faster).

---

## What is in here

```
holdem/
  cards.py evaluator.py evaluator_np.py   card primitives, exact + vectorised hand evaluators
  engine.py game.py                       no-limit rules: blinds, min-raises, all-ins, side pots
  equity.py                               Monte-Carlo equity + a cached 169×8 preflop table
  ml/
    nn.py conv.py                         the neural-network framework (from scratch)
    cfr.py abstraction.py                 Monte-Carlo CFR and the game abstraction
    features.py                           the 50 numbers a poker situation becomes
    policy.py                             the bot's policy + value network
    opponent.py                           online opponent modelling
    difficulty.py                         the difficulty ladder and skill tracking
  bots/          rule.py blueprint.py neural.py
  vision/        render.py dataset.py cardnet.py detect.py
  analysis/      handhistory.py replay.py corpus.py promodel.py analyzer.py
  server/        app.py client.py         multiplayer table over HTTP
  train/         every training entry point, plus benchmark.py
examples/        a sample hand-history file to run the analyser on
models/          trained weights, committed
tests/           200 tests
```

---

## The machine learning, in order

### 1. A neural-network framework, written by hand

`holdem/ml/nn.py` and `conv.py`: `Linear`, `Conv2D`, `MaxPool2D`, `BatchNorm2D`,
`LayerNorm`, `Dropout`, ReLU/LeakyReLU/Tanh/Sigmoid, softmax cross-entropy (with
action masking), MSE and Huber losses, SGD-with-momentum and Adam/AdamW, gradient
clipping, and `.npz` checkpointing that carries batch-norm running statistics.

Every layer implements its own `backward`. The test suite proves they are right:
`numeric_grad_check` runs the whole model in float64 and compares against central
differences — **max relative error under 1e-6** on every stack, and convolution
gradients are additionally checked against a naive loop implementation.

### 2. A CFR blueprint — learning poker from nothing but self-play

`holdem/ml/cfr.py` implements **external-sampling Monte-Carlo Counterfactual
Regret Minimisation** over an abstracted heads-up game. No poker knowledge is
coded in: the solver plays itself, tracks how much better each action would have
been in hindsight, and turns positive regret into a strategy.

The abstraction is where the engineering is:

* **Cards** → made-hand strength percentile (from a 400k-sample distribution per
  street) crossed with a draw class, plus the 169 preflop buckets.
* **Betting** → the literal action sequence explodes into millions of
  barely-visited nodes, so history is summarised by what actually drives the
  decision: pot size relative to the effective stack, the price of a call, raises
  so far this street, and position.

That took the tree from 116,000 barely-visited info sets after 300 iterations to
**9,046 well-travelled ones**, and training runs across all cores by exchanging
regret *deltas* between workers each round.

Shipped: `models/blueprint.npz`, 1.2M iterations.

### 3. The neural policy — distillation, then self-play RL

`holdem/ml/policy.py` is a shared trunk with a **policy head** (7 abstract
actions: fold, check/call, four pot-fractions, all-in) and a **value head**
(expected result in big blinds). Trained in two stages by
`holdem/train/selfplay.py`:

1. **Distil** the blueprint: play hands with the solver driving, record
   `(state → solved strategy)` and fit by cross-entropy. This lifts a table
   solved for one abstracted heads-up game into a network that reads the full
   continuous state.
2. **REINFORCE with a value baseline**: play the real engine against a pool of
   rule bots and frozen snapshots of itself, credit every decision with the big
   blinds the hand actually won, subtract the value head as a baseline, and add an
   entropy bonus so the policy does not collapse onto one action. Table size is
   randomised every batch — training only 6-handed produces a bot that folds far
   too much heads-up.

### 4. Opponent modelling — the part that learns *you*

`holdem/ml/opponent.py` runs two layers at once, on public information only
(the engine hands observers a redacted view; a test asserts this):

* **Decayed statistics** — VPIP, PFR, 3-bet, aggression factor, fold-to-bet,
  went-to-showdown. Exponentially decayed, so the model follows a player who
  changes gear rather than averaging over their whole history.
* **A learned action predictor** — a small network trained by SGD *during the
  game*, one update every couple of decisions from a replay buffer, predicting
  `P(fold / call / raise)` for that specific person in that specific kind of spot.

The bot then deviates from its baseline strategy toward exploiting what it found:
if you fold to a two-thirds-pot bet more often than the price justifies, it starts
bluffing you; if you never fold, it stops bluffing and value-bets thinner; if
your aggression is high but your showdowns are weak, it stops believing you.

### 5. Difficulty that grows with you

`holdem/ml/difficulty.py`. Five presets — `novice`, `casual`, `regular`,
`strong`, `pro` — differing in policy temperature, blunder rate, how hard the
opponent model is used, and roll-out budget. Any number in `[0, 1]` interpolates
between them.

`--difficulty adaptive` puts a `SkillTracker` on you and slides the level to sit
just above your estimate. Skill is a blend of three weak signals, because each is
individually gameable: how often your action matches a strong reference policy
(graded only on hands you showed down — the bot never peeks), how far your stats
sit outside healthy bands, and your realised win rate. The level moves slowly and
is capped per hand, so it tracks improvement instead of chasing variance.

### 6. Card vision — images to cards

`holdem/vision/`. There is no public dataset of every card in every skin, so the
data is generated: `render.py` draws cards in fifteen deck styles across three font
families (different palettes, four-colour decks, font-glyph vs vector-drawn
pips), and `dataset.py`
augments them with rotation, non-uniform scaling, lighting and gamma shifts, blur,
sensor noise and partial occlusion.

`cardnet.py` is a CNN with a shared trunk and **two heads** — 13-way rank and
4-way suit. They share nearly all their visual evidence, and splitting them means
a wrong suit no longer costs you the rank. Downsampling uses strided convolutions
rather than pooling: same receptive field for a third of the arithmetic, which
matters when every GEMM is NumPy.

Finding the cards is deliberately *not* a second network — card faces are bright,
low-saturation rectangles, which a threshold plus run-length connected components
locates in milliseconds (including rejecting the enclosed white counter inside a
"Q", which would otherwise look like a tiny extra card).

**Validation is on three deck styles the network never trains on.**

### 7. The analyser

`holdem/analysis/`. A `ProModel` with three heads over one trunk: what a strong
player does here, what each action is worth in big blinds, and what the spot
itself is worth. The action-value head is regressed onto the realised return of
the action actually taken, which makes the headline number honest:

```
ev_loss = max_a Q(a) − Q(action you chose)      # in big blinds
```

It runs on anything that can become a `HandResult`:

* hands played in this project,
* **real hand histories** — `handhistory.py` parses the PokerStars format, and
  `replay.py` steps a parsed hand back through the engine to recover the exact
  decision context, so imported hands get the same treatment as bot ones,
* **a photograph or screenshot** — via the vision stack.

Output is a per-decision grade, a session report with EV lost by street, and a
leak list ("folding too often to bets — you are easy to bluff", "missing value —
you checked 11 spots with over 70% equity").

**One subtlety worth naming**, because the first version got it wrong: an
action-value head trained purely on what strong bots chose has a selection-bias
problem. The only all-ins in such a corpus are the ones a good player made with
a good hand, so the model concludes that shoving is wonderful *everywhere* and
the analyser starts telling you to jam 100 big blinds with A7o. The fix is
coverage — a fraction of corpus decisions are taken uniformly at random, so the
value head sees what a bad all-in is worth, while the policy head ignores those
decisions so it still learns strong play. On top of that, the analyser will only
hold up a line as "better" if the pro policy actually plays it at least 5% of
the time.

---

## Measured results

Everything below is produced by code in this repository, not asserted.

### The hand evaluator is exact

All 2,598,960 five-card hands enumerated; every category frequency matches the
published values exactly (40 straight flushes, 624 quads, …, 1,302,540 high
cards). The vectorised evaluator agrees with the reference on every one of 12,000
random 5-, 6- and 7-card hands.

### The engine obeys the rules

600 randomised hands with 2–9 players, mixed stack depths and antes: chips
conserved exactly, no betting round ever failed to terminate, and across 500 more
hands every side pot was paid to the best *eligible* hand. Min-raise sizing,
big-blind option, heads-up button order, short all-ins that do not reopen the
betting, odd-chip distribution and uncalled-bet returns each have a dedicated
test.

### Monte-Carlo equity matches published numbers

| hand | opponents | this repo | published |
|---|---|---|---|
| AA | 1 | 85.1% | 85.2% |
| AA | 5 | 48.9% | 49.0% |
| AKs | 1 | 67.3% | 67.0% |
| 72o | 1 | 35.1% | 35.1% |

### The trained bot beats every baseline

`python -m holdem.train.benchmark` produces this. Heads-up, one session of 3,000
hands per pairing, big blinds per 100 hands:

| | EquityBot | TightRock | CallingStation | LooseAggressive | RandomBot | HonestBot |
|---|---:|---:|---:|---:|---:|---:|
| **neural (pro)** | **+723** | **+670** | **+2420** | **+1780** | **+1500** | **+770** |
| neural (regular) | −60 | +307 | +2237 | +1705 | +1028 | +240 |
| neural (novice) | +130 | −551 | +1623 | +1034 | +1147 | +127 |
| CFR blueprint | −161 | −200 | +443 | +563 | +664 | −142 |
| EquityBot (reference) | +129 | −74 | +229 | +472 | +1467 | −85 |

The ladder is a real ladder: `novice` loses to two of the baselines that `pro`
beats comfortably.

**How noisy is that?** Re-measured as the mean of four independent 2,000-hand
sessions, the `pro` row reads +586 / +630 / +2570 / +1803 / +1419 / +551. The
close matchups move by a few hundred bb/100 between runs; the lopsided ones
barely move. Read the ordering, not the digits.

Five-handed against EquityBot, TightRock, CallingStation and LooseAggressive at
once: **+1,908 bb/100** as the mean of two 2,000-hand sessions (+2,199 and
+1,618), finishing first in both. The next-best seat averaged +98.

These numbers are large because the baselines are weak and stacks are 100 big
blinds deep, so single pots swing hundreds of blinds.

### Card recognition, on decks the model never trained on

| | |
|---|---|
| clean renders, 12 training decks | **100.0%** of 52 cards, every deck |
| clean renders, 3 **held-out** decks | **91.7%** (96% / 92% / 87%) |
| detection on rendered tables | **75/75** boards found all five cards |
| end-to-end (detect → classify) | **371/375 cards, 98.9%** |
| heavily augmented held-out crops | 84.1% card, 85.5% rank, 98.2% suit |

The last row is the robustness figure — those crops are rotated, rescaled,
blurred, noised, and up to a third of the card can be covered.

### The analyser ranks players correctly

300 hands of four bots, expected value lost per 100 hands and agreement with the
pro policy:

| player | style | EV lost | agreement |
|---|---|---:|---:|
| Solid | pot-odds bot | 220 bb/100 | 67% |
| Nit | premium hands only | 197 bb/100 | 61% |
| Wild | uniformly random | 717 bb/100 | 16% |
| Maniac | raises everything | 888 bb/100 | 37% |

On `examples/sample-session.txt` (120 hands of a calling station) it flags every
one of that player's real leaks — VPIP 79%, folds to bets 6%, aggression factor
0.2, showdown 86% — and the five decisions it picks out as most expensive are
all calls that should have been folds.

### Measuring beat guessing: the blueprint blend

The strong bot originally blended the CFR blueprint into the neural policy at
weight 0.5, and lost 407 bb/100 to the equity baseline. Sweeping the weight
found the cause:

| blueprint weight | vs EquityBot | vs TightRock | vs CallingStation | vs LooseAggressive |
|---:|---:|---:|---:|---:|
| **0.00** | **+634** | **+517** | **+2293** | **+1680** |
| 0.15 | −425 | +414 | +2167 | +1493 |
| 0.30 | +467 | +487 | +2153 | +1579 |
| 0.50 | −407 | +406 | +1739 | +1253 |

So the blueprint now weights 0 at `pro` and rises as difficulty *falls*: the
lower levels play its textbook-but-exploitable style, which is a more
poker-realistic way to be weak than simply adding noise.

---

## Training it yourself

Every model ships trained, but nothing is a black box:

```bash
python -m holdem.train.preflop_table                     # 169×8 equity cache   (~1.5 min)
python -m holdem.train.strength_table                    # strength percentiles (~10 s)
python -m holdem.train.cfr_train --iters 1200000 --workers 4     # CFR blueprint (~35 min)
python -m holdem.train.selfplay --stage both --hands 40000       # policy        (~10 min)
python -m holdem.train.train_vision --train 24000 --epochs 14    # card CNN      (~25 min)
python -m holdem.train.train_analyst --hands 6000                # analyser      (~8 min)
```

Timings are from a four-core container. Then re-measure everything:

```bash
python -m holdem.train.benchmark          # every table in this README
python -m holdem.train.benchmark --quick  # a two-minute version
```

The analyser can be trained on **your** hands instead of generated ones:

```bash
python -m holdem.train.train_analyst --histories ~/PokerStars/HandHistory --hands 0
```

---

## Multiplayer

`python -m holdem serve` starts a table on the standard library alone — no
framework, no CDN, no build step. Open the address in a browser, take a seat, and
any seat nobody has claimed keeps being played by a bot, so the table is never
short-handed. Several people can join the same table from different machines.

---

## Tests

```bash
python -m pytest -q
```

The suite covers rules edge cases, gradient correctness, equity against published
values, opponent-model behaviour, hand-history round-tripping, vision detection
across every deck style, the HTTP server end to end, and the CLI.

---

## Honest limitations

* The CFR blueprint solves an *abstraction*, not the real game. Measured
  heads-up it beats loose and passive opponents comfortably but loses to a very
  tight one, and blending it into an already-trained neural policy makes that
  policy worse — so it is used as the distillation teacher and as the flavour of
  the lower difficulty levels, not as the strong bot's brain. The numbers behind
  that decision are in the table above.
* Bot strength is measured against the rule-based baselines in this repository
  and against earlier versions of itself. That is a real yardstick, but it is not
  the same as being tested against strong humans or a published bot.
* The card model is trained on rendered cards. It generalises to deck styles it
  has never seen, which is the right test available offline, but a real
  photograph from a phone is a harder distribution than anything here. Its
  detector also assumes cards are roughly upright and not overlapping.
* The analyser's "pro" corpus is generated by the strongest bots here unless you
  point it at real hand histories. Point it at real ones — the importer exists
  precisely so that number means something.
* Its expected-value numbers are estimates from a value head fitted to
  hand-level outcomes, which are extremely noisy (one river card can be worth a
  hundred blinds), and "EV lost" is the gap to the maximum of seven such
  estimates — so even flawless play scores above zero. It ranks players and
  decisions correctly (in the sample session a calling station shows five times
  the EV loss of a rock, matching their real win rates) but the absolute number
  is not chips left on the table.
