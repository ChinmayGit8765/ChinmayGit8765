# Why the contribution graph is nearly empty

Measured 2026-08-30 across every repository on the `ChinmayGit8765` account,
public and private, default branches only (the only branches GitHub counts).

## August 2026 — 156 commits, 6 of them counted

| author email | GitHub credits | commits | on the graph |
| --- | --- | ---: | :---: |
| `cpur0011@student.monash.edu` | **@chinmayuni12371** | 88 | no |
| `actions@users.noreply.github.com` | nobody | 30 | no |
| `chinmaypurohit1010@gmail.com` | nobody | 15 | no |
| `noreply@anthropic.com` | @claude | 8 | no |
| `41898282+github-actions[bot]@…` | the Actions app | 7 | no |
| `careers.chinmay@gmail.com` | **@ChinmayGit8765** | 6 | **yes** |

**3.8% of the month's commits landed on the graph.** The rest are real commits
in real repositories — they are simply signed with an email GitHub reads as
somebody else.

All time, the leak is larger: **2,337** commits authored with
`cpur0011@student.monash.edu` are credited to `@chinmayuni12371`, against
**32** ever credited to `@ChinmayGit8765`.

## The rule GitHub applies

A commit counts toward a profile's contribution graph only when *all* of these
hold:

1. the commit's **author email** is a **verified email on that account**;
2. the commit sits on the repository's **default branch** (or `gh-pages`);
3. the repository is not a fork, and is owned by the account or one it
   collaborates on.

Nothing else matters — not who pushed, not who owns the repo, not how much work
is in the diff. An unrecognised author email is worth zero, silently.

## Fixes

### 1. Consolidate the student address onto @ChinmayGit8765

This is the step that reclaims 2,337 commits, and it is the one step no script
can do — GitHub verifies an address by mailing it, so it needs your inbox.
An address can be verified on **one account only**, so this is a move, not a
copy. Do it in this order:

1. **@chinmayuni12371** → [Settings → Emails](https://github.com/settings/emails)
   → delete `cpur0011@student.monash.edu`. Adding it to the other account first
   will simply fail while it is still held here.
2. **@ChinmayGit8765** → [Settings → Emails](https://github.com/settings/emails)
   → *Add email address* → `cpur0011@student.monash.edu` → open the
   verification mail and confirm.
3. Same again for `chinmaypurohit1010@gmail.com`. It is verified on no account
   at all, so there is nothing to remove first — just add and verify.
4. Add both to the `ALLOWED` list in `scripts/consolidate_identity.sh` and
   re-run it, so the guard stops treating them as leaks.
5. A day later: `python3 scripts/attribution_audit.py --days 30` — the Monash
   row should now read `ChinmayGit8765` / `yes`.

GitHub re-attributes historical commits the moment an address is verified and
backfills the contribution graph, so the past year of Monash-signed work
reappears. Give it up to ~24 hours to redraw.

Three things worth knowing before you start:

- **Do it while the Monash inbox still works.** Verification needs a live
  mailbox. After you graduate the address is unrecoverable, and with it the
  2,337 commits.
- **Removing a verified address de-attributes its commits again.** Once moved,
  leave it on the account — it costs nothing to keep, and it is what holds the
  history in place.
- **If "Block command line pushes that expose my email" is on**, pushes
  authored with `careers.chinmay@gmail.com` are rejected. Either turn that off,
  or switch the identity everywhere to
  `193141422+ChinmayGit8765@users.noreply.github.com`, which counts identically
  and is already in the allowlist.

Rather keep the accounts separate? Then step 3 alone still recovers 15 commits
a month, and every future commit must use `careers.chinmay@gmail.com`.

### 2. Stop the leak at the source (every machine, every agent)

`scripts/consolidate_identity.sh` does all three parts of this. It is a dry run
until you pass `--apply`, and is safe to run repeatedly:

```sh
./scripts/consolidate_identity.sh              # show the plan
./scripts/consolidate_identity.sh --apply      # global identity, guard, sweep
./scripts/consolidate_identity.sh --apply --root ~/code --depth 6
```

It sets the global identity, installs a `pre-commit` guard in a global
`core.hooksPath` that refuses any commit whose author email is not verified on
the account, and sweeps every checkout under `--root` for stale local
overrides. Repos that set `core.hooksPath` themselves — husky and friends —
win over the global setting and are left alone, so the guard cannot displace a
project's own hooks.

Agent sessions are the sharp edge. A Claude Code cloud container ships with
`user.email=noreply@anthropic.com`, which is why 8 commits in August went to
`@claude`, and no per-machine setting reaches inside a fresh container. So this
repo also carries `.claude/hooks/session-start.sh`, registered in
`.claude/settings.json`: every session, cloud or local, pins the checkout's
identity before anything can commit. Copy those two files into any repo an
agent touches — they are eleven lines and have no dependencies.

### 3. Decide what the cron bots commit as

The 37 bot commits a month are the daily digest, the pirate picker, the AFL
sweep and the market feed — automation you wrote, committing generated data.
They currently count for nobody. Pointing them at your verified email would put
them on the graph; leaving them as bots keeps the graph a record of hand-written
work. Both are defensible. What is not defensible is the current state being an
accident rather than a choice.

### 4. Push more of the work you already do

Even fully attributed, August is 156 commits over 30 days — about 5 a day, not
40. Two structural reasons, both worth knowing:

- **Only default-branch commits count.** Work sitting on an unmerged `claude/*`
  branch contributes nothing until it lands on `main`. A squash merge collapses
  a 40-commit branch into a single counted commit; a merge commit preserves all
  40. If the fleet's branches are squashed, that alone is the difference between
  the graph you expect and the one you see.
- **Unpushed worktrees are invisible.** Commits in a local worktree count for
  nothing until they reach GitHub.

## Keeping it honest

`scripts/attribution_audit.py` tallies any window by author email and by the
account GitHub credited, so this never has to be re-derived by hand:

```sh
python3 scripts/attribution_audit.py --days 30
GITHUB_TOKEN=… python3 scripts/attribution_audit.py --days 7 --fail-under 90
```

`.github/workflows/attribution-audit.yml` runs it weekly and fails the job when
under 90% of the week's commits are credited to `@ChinmayGit8765`. Add an
`ATTRIBUTION_TOKEN` secret (classic PAT, `repo` scope) to include private
repositories; without it the audit sees public repositories only.
