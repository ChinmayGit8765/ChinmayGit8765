#!/usr/bin/env bash
# Consolidate every commit identity on this machine onto one GitHub account.
#
# Adding the student address to @ChinmayGit8765 reclaims the past. This keeps
# the future consolidated: one global identity, a pre-commit guard that refuses
# an email GitHub would credit to somebody else, and a sweep that repairs
# checkouts already carrying a stale local override.
#
#   ./scripts/consolidate_identity.sh                 # dry run — shows the plan
#   ./scripts/consolidate_identity.sh --apply         # do it
#   ./scripts/consolidate_identity.sh --apply --root ~/code
#   ./scripts/consolidate_identity.sh --apply --strict   # force ONE address
#
# Safe to run repeatedly.

set -euo pipefail

NAME="Chinmay Purohit"
# GitHub's noreply alias for @ChinmayGit8765: always counts, never exposes
# the real inbox, and passes the "block pushes that expose my email" setting.
EMAIL="193141422+ChinmayGit8765@users.noreply.github.com"
# Addresses GitHub has VERIFIED on @ChinmayGit8765 today. A commit authored
# with anything else is worth zero contributions, so anything not listed here
# gets rewritten to $EMAIL.
#
# Add an address here only once Settings -> Emails shows it verified on this
# account — listing it early lets commits keep leaking to wherever it is
# verified now. Once cpur0011@student.monash.edu and
# chinmaypurohit1010@gmail.com are moved over, add them and re-run.
ALLOWED=(
  "193141422+ChinmayGit8765@users.noreply.github.com"
  "careers.chinmay@gmail.com"
)
ROOT="$HOME"
DEPTH=5
HOOKS_PATH="$HOME/.config/git/hooks"
APPLY=0
GUARD=1
STRICT=0

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --no-guard) GUARD=0 ;;
    --strict) STRICT=1 ;;
    --root) ROOT="$2"; shift ;;
    --depth) DEPTH="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --email) EMAIL="$2"; shift ;;
    --hooks-path) HOOKS_PATH="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

run() {
  if [ "$APPLY" -eq 1 ]; then
    "$@"
  else
    printf '  would run: %s\n' "$*"
  fi
}

is_allowed() {
  local candidate="$1" allowed
  for allowed in "${ALLOWED[@]}"; do
    [ "$candidate" = "$allowed" ] && return 0
  done
  return 1
}

[ "$APPLY" -eq 1 ] || echo "DRY RUN — nothing is written. Re-run with --apply."

echo
echo "1. Global identity"
current_email="$(git config --global user.email || true)"
if [ "$current_email" = "$EMAIL" ]; then
  echo "  already $NAME <$EMAIL>"
else
  echo "  ${current_email:-(unset)} -> $EMAIL"
  run git config --global user.name "$NAME"
  run git config --global user.email "$EMAIL"
fi

if [ "$GUARD" -eq 1 ]; then
  echo
  echo "2. Pre-commit guard at $HOOKS_PATH"
  existing_hooks_path="$(git config --global core.hooksPath || true)"
  if [ -n "$existing_hooks_path" ] && [ "$existing_hooks_path" != "$HOOKS_PATH" ]; then
    echo "  WARNING: global core.hooksPath is already $existing_hooks_path — leaving it alone."
    echo "  Add the guard to that directory by hand, or pass --hooks-path $existing_hooks_path."
  else
    if [ "$APPLY" -eq 1 ]; then
      mkdir -p "$HOOKS_PATH"
      {
        echo '#!/bin/sh'
        echo '# Refuse a commit GitHub would credit to somebody else.'
        printf 'allowed="%s"\n' "${ALLOWED[*]}"
        cat <<'GUARD'
email=$(git config user.email)
case " $allowed " in
  *" $email "*) exit 0 ;;
esac
echo "pre-commit: author email '$email' is not verified on @ChinmayGit8765," >&2
echo "so this commit would not count. Fix it with:" >&2
echo "  git config user.email 193141422+ChinmayGit8765@users.noreply.github.com" >&2
exit 1
GUARD
      } > "$HOOKS_PATH/pre-commit"
      chmod +x "$HOOKS_PATH/pre-commit"
      git config --global core.hooksPath "$HOOKS_PATH"
      echo "  installed, and core.hooksPath now points at it"
      echo "  note: repos with their own hooks (husky and friends) set core.hooksPath"
      echo "  locally, and a local setting still wins — those are unaffected."
    else
      echo "  would write $HOOKS_PATH/pre-commit and set global core.hooksPath"
    fi
  fi
fi

echo
if [ "$STRICT" -eq 1 ]; then
  echo "3. Checkouts under $ROOT not on $EMAIL (--strict)"
else
  echo "3. Checkouts under $ROOT with a stale local identity"
fi
found=0
while IFS= read -r gitdir; do
  repo="$(dirname "$gitdir")"
  local_email="$(git -C "$repo" config --local user.email || true)"
  [ -n "$local_email" ] || continue
  if [ "$STRICT" -eq 1 ]; then
    [ "$local_email" = "$EMAIL" ] && continue
  else
    is_allowed "$local_email" && continue
  fi
  found=$((found + 1))
  echo "  $repo: $local_email -> $EMAIL"
  run git -C "$repo" config --local user.name "$NAME"
  run git -C "$repo" config --local user.email "$EMAIL"
done < <(find "$ROOT" -maxdepth "$DEPTH" -name .git -print 2>/dev/null)

[ "$found" -eq 0 ] && echo "  none — every local override already counts"

if [ "$STRICT" -eq 0 ]; then
  echo
  echo "careers.chinmay@gmail.com still counts and is left alone. If you turn on"
  echo "\"block command line pushes that expose my email\", re-run with --strict to"
  echo "move every checkout onto the noreply address."
fi

echo
echo "Verify with:  python3 scripts/attribution_audit.py --days 7"
