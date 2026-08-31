#!/bin/bash
# Pin the commit identity for this checkout.
#
# A Claude Code cloud container ships with user.email=noreply@anthropic.com, so
# any commit an agent session makes is credited to @claude and is worth zero
# contributions on this account. This hook sets the identity per checkout, at
# session start, before anything can commit.
#
# The address is GitHub's noreply alias for this account: it always counts,
# it keeps the real inbox out of public history, and it survives the
# "block command line pushes that expose my email" setting.
#
# Local sessions inherit whatever the machine's global identity is — run
# scripts/consolidate_identity.sh once per machine to make that correct too.

set -euo pipefail

AUTHOR_NAME="Chinmay Purohit"
AUTHOR_EMAIL="193141422+ChinmayGit8765@users.noreply.github.com"

cd "${CLAUDE_PROJECT_DIR:-$PWD}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not a git checkout — commit identity not set."
  exit 0
fi

git config user.name "$AUTHOR_NAME"
git config user.email "$AUTHOR_EMAIL"

echo "Commit identity for this checkout: $AUTHOR_NAME <$AUTHOR_EMAIL> (verified on @ChinmayGit8765, so commits count)."
