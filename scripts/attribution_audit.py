#!/usr/bin/env python3
"""Audit which of my commits actually land on my GitHub contribution graph.

GitHub only counts a commit toward a profile's contribution graph when the
commit's *author email* is a verified email on that account, and the commit
sits on a repository's default branch (or gh-pages). Commits authored with any
other email — a university address, a bot identity, a tool default — are worth
exactly zero contributions, no matter how much work is in them.

This script walks the commit search API over an owner's repositories and tallies
every commit in a window by author email and by the account GitHub credited it
to, so the leak is visible as a number instead of a vibe.

Usage:
    python3 scripts/attribution_audit.py --days 30
    python3 scripts/attribution_audit.py --since 2026-08-01 --until 2026-08-31
    python3 scripts/attribution_audit.py --days 7 --fail-under 90   # for CI

Auth: set GITHUB_TOKEN to include private repositories and raise the rate
limit. Without a token only public repositories are visible, so the numbers
undercount.

Stdlib only — no dependencies, runs anywhere Python 3.9+ does.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
PER_PAGE = 100
MAX_PAGES = 10  # GitHub search caps results at 1000


def iso_date(value: str) -> str:
    """Validate a YYYY-MM-DD date and return it unchanged."""
    dt.date.fromisoformat(value)
    return value


def request(url: str, token: str | None, attempt: int = 0) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "attribution-audit")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        # Secondary rate limits on the search API are common and transient.
        if err.code in (403, 429) and attempt < 4:
            time.sleep(2 ** attempt * 5)
            return request(url, token, attempt + 1)
        body = err.read().decode("utf-8", "replace")[:300]
        raise SystemExit(f"GitHub API {err.code} for {url}\n{body}") from err


def search_commits(owner: str, since: str, until: str, token: str | None) -> list[dict]:
    """Every commit on default branches of `owner`'s repos in the window."""
    query = f"user:{owner} author-date:{since}..{until}"
    commits: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        params = urllib.parse.urlencode(
            {"q": query, "per_page": PER_PAGE, "page": page, "sort": "author-date"}
        )
        payload = request(f"{API}/search/commits?{params}", token)
        items = payload.get("items", [])
        commits.extend(items)
        if len(items) < PER_PAGE:
            break
        if len(commits) >= payload.get("total_count", 0):
            break
    return commits


def credited_login(item: dict) -> str | None:
    """The account GitHub credited, or None when the email matches nobody."""
    author = item.get("author")
    return author.get("login") if author else None


def audit(owner: str, since: str, until: str, token: str | None) -> dict:
    commits = search_commits(owner, since, until, token)

    by_email: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    by_day: collections.Counter = collections.Counter()
    credited_by_day: collections.Counter = collections.Counter()

    for item in commits:
        commit = item.get("commit", {})
        email = commit.get("author", {}).get("email", "(unknown)")
        login = credited_login(item)
        by_email[email][login or "(nobody)"] += 1

        day = commit.get("author", {}).get("date", "")[:10]
        if day:
            by_day[day] += 1
            if login and login.lower() == owner.lower():
                credited_by_day[day] += 1

    total = len(commits)
    credited = sum(
        count
        for email, logins in by_email.items()
        for login, count in logins.items()
        if login.lower() == owner.lower()
    )

    return {
        "owner": owner,
        "since": since,
        "until": until,
        "total_commits": total,
        "credited_to_owner": credited,
        "credited_pct": round(100 * credited / total, 1) if total else 0.0,
        "by_email": {
            email: dict(logins.most_common()) for email, logins in by_email.items()
        },
        "commits_per_day": dict(sorted(by_day.items())),
        "credited_per_day": dict(sorted(credited_by_day.items())),
    }


def render(result: dict) -> str:
    owner = result["owner"]
    total = result["total_commits"]
    lines = [
        f"# Commit attribution — {owner}",
        "",
        f"Window: {result['since']} .. {result['until']}",
        f"Commits found on default branches: **{total}**",
        f"Credited to @{owner}: **{result['credited_to_owner']}** "
        f"({result['credited_pct']}%)",
        "",
        "| author email | credited to | commits | counts on graph |",
        "| --- | --- | ---: | :---: |",
    ]

    rows = [
        (email, login, count)
        for email, logins in result["by_email"].items()
        for login, count in logins.items()
    ]
    for email, login, count in sorted(rows, key=lambda row: -row[2]):
        counts = "yes" if login.lower() == owner.lower() else "no"
        lines.append(f"| `{email}` | {login} | {count} | {counts} |")

    days = result["commits_per_day"]
    if days:
        span = len(days)
        lines += [
            "",
            f"Active days: {span} · commits/day: "
            f"{total / span:.1f} pushed, "
            f"{result['credited_to_owner'] / span:.1f} on the graph",
        ]

    leaks = sorted(
        (
            (email, login, count)
            for email, login, count in rows
            if login.lower() != owner.lower()
        ),
        key=lambda row: -row[2],
    )
    if leaks:
        lines += ["", "## Leaking identities", ""]
        for email, login, count in leaks:
            where = "no GitHub account" if login == "(nobody)" else f"@{login}"
            plural = "commit" if count == 1 else "commits"
            lines.append(f"- `{email}` → {where} — {count} {plural} lost")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="ChinmayGit8765", help="GitHub account")
    parser.add_argument("--days", type=int, default=30, help="window ending today")
    parser.add_argument("--since", type=iso_date, help="YYYY-MM-DD, overrides --days")
    parser.add_argument("--until", type=iso_date, help="YYYY-MM-DD, defaults to today")
    parser.add_argument("--json", action="store_true", help="emit JSON, not markdown")
    parser.add_argument(
        "--fail-under",
        type=float,
        help="exit 1 when the credited percentage falls below this",
    )
    args = parser.parse_args()

    until = args.until or dt.date.today().isoformat()
    since = args.since or (
        dt.date.fromisoformat(until) - dt.timedelta(days=args.days)
    ).isoformat()

    result = audit(args.owner, since, until, os.environ.get("GITHUB_TOKEN"))
    print(json.dumps(result, indent=2) if args.json else render(result))

    if args.fail_under is not None and result["total_commits"]:
        if result["credited_pct"] < args.fail_under:
            print(
                f"\nFAIL: {result['credited_pct']}% credited, "
                f"below the {args.fail_under}% floor",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
