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

### 1. Reclaim the 2,337 commits (one time, ~5 minutes)

An email address can be verified on only one GitHub account, so this is a move,
not a copy:

1. Sign in as **@chinmayuni12371** → Settings → Emails → remove
   `cpur0011@student.monash.edu`.
2. Sign in as **@ChinmayGit8765** → Settings → Emails → add
   `cpur0011@student.monash.edu` → confirm the verification mail.
3. Do the same for `chinmaypurohit1010@gmail.com`, which is verified nowhere
   and is losing commits to no one at all.

GitHub re-attributes historical commits when an email is added, and backfills
the contribution graph — the past year of Monash-signed work reappears.

Rather keep the accounts separate? Then step 3 alone still recovers 15 commits
a month, and every future commit must use `careers.chinmay@gmail.com`.

### 2. Stop the leak at the source (every machine, every agent)

Set the identity globally, once per machine:

```sh
git config --global user.name "Chinmay Purohit"
git config --global user.email "careers.chinmay@gmail.com"
```

Agent sessions are the sharp edge here. A Claude Code cloud container ships
with `user.email=noreply@anthropic.com`, which is why 8 commits in August are
credited to `@claude`. Any agent, worktree, or CI checkout that commits on your
behalf needs the identity set explicitly inside that checkout.

To make a repository refuse a wrong-identity commit before it is written, drop
this in `.git/hooks/pre-commit` (or a shared `core.hooksPath` directory):

```sh
#!/bin/sh
allowed="careers.chinmay@gmail.com cpur0011@student.monash.edu"
email=$(git config user.email)
case " $allowed " in
  *" $email "*) ;;
  *) echo "refusing: author email '$email' does not count on your graph" >&2
     exit 1 ;;
esac
```

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
