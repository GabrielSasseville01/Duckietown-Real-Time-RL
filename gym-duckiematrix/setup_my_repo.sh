#!/bin/bash
# Script to set up your own repository and push to it

# USAGE:
# 1. First, create a new repository on GitHub:
#    - Go to: https://github.com/new
#    - Choose a name (e.g., "gym-duckiematrix" or "duckiematrix-rl")
#    - Choose Public or Private
#    - DO NOT initialize with README/license
# 2. Replace YOUR_REPO_NAME below with your actual repository name
# 3. Run this script: bash setup_my_repo.sh

YOUR_REPO_NAME="YOUR_REPO_NAME_HERE"  # CHANGE THIS to your actual repo name
GITHUB_USER="GabrielSasseville01"

if [ "$YOUR_REPO_NAME" = "YOUR_REPO_NAME_HERE" ]; then
    echo "ERROR: Please edit this script and set YOUR_REPO_NAME to your actual repository name"
    echo ""
    echo "Steps:"
    echo "1. Create repo at: https://github.com/new"
    echo "2. Edit this script and replace YOUR_REPO_NAME_HERE with your repo name"
    echo "3. Run this script again"
    exit 1
fi

REPO_URL="git@github.com:${GITHUB_USER}/${YOUR_REPO_NAME}.git"

echo "Setting up repository: $REPO_URL"
echo ""

# Add your repo as a new remote called "myrepo"
echo "Adding remote 'myrepo'..."
git remote add myrepo "$REPO_URL" 2>/dev/null || git remote set-url myrepo "$REPO_URL"

echo ""
echo "Current remotes:"
git remote -v

echo ""
echo "Pushing all branches to your repository..."
echo "This will push:"
git branch --list | sed 's/^/  - /'

echo ""
read -p "Continue with push? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Push all branches
git push -u myrepo --all

# Push tags if any
git push -u myrepo --tags

echo ""
echo "✅ Done! Your repository is available at:"
echo "   https://github.com/${GITHUB_USER}/${YOUR_REPO_NAME}"
echo ""
echo "To push future changes:"
echo "   git push myrepo ente"

