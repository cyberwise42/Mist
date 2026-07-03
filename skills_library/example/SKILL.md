---
name: git-workflow
description: Steps for common git operations like status, commit, branch, and push
---

# Git workflow

Use the `shell` tool for all git operations. Follow these steps exactly.

## Check status
1. Run `git status --short` to see changed files.
2. Run `git log --oneline -5` to see recent commits.

## Commit changes
1. Run `git status --short` first — never commit blind.
2. Run `git add <specific files>` (avoid `git add .` unless asked).
3. Run `git commit -m "<imperative, under 72 chars>"`.

## Create a branch
1. Run `git checkout -b <branch-name>`.
2. Branch names: lowercase, hyphens, prefixed with `fix/` or `feat/`.

## Push
1. Run `git push -u origin <branch>` on first push of a branch.
2. Report the exact output back to the user, including any errors.
