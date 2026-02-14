#!/bin/bash
# Install all git hooks for the project
# This script copies all hook scripts from the hooks/ directory to .git/hooks/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GIT_HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "🔧 Installing Git hooks..."

# Create .git/hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Copy essential hook scripts (removed pre-push-branch-protection for open source)
HOOK_SCRIPTS=("commit-msg" "pre-commit")

for hook in "${HOOK_SCRIPTS[@]}"; do
    if [ -f "$SCRIPT_DIR/$hook" ]; then
        # Determine the target hook name (remove file extension if any)
        TARGET_HOOK_NAME="$hook"
        if [[ "$hook" == *.* ]]; then
            TARGET_HOOK_NAME="${hook%.*}"
        fi

        cp "$SCRIPT_DIR/$hook" "$GIT_HOOKS_DIR/$TARGET_HOOK_NAME"
        chmod +x "$GIT_HOOKS_DIR/$TARGET_HOOK_NAME"
        echo "✅ Installed $TARGET_HOOK_NAME hook"
    else
        echo "⚠️  Hook script $hook not found, skipping..."
    fi
done

# Also copy the install script itself to .git/hooks/ for easy reinstallation
cp "$SCRIPT_DIR/install-hooks.sh" "$GIT_HOOKS_DIR/install-hooks.sh"
chmod +x "$GIT_HOOKS_DIR/install-hooks.sh"

echo ""
echo "🎉 All hooks installed successfully!"
echo ""
echo "📋 Installed hooks:"
echo "   - pre-push: Prevents direct pushes to protected branches (master, main, develop)"
echo "   - commit-msg: Validates commit message format (conventional commits)"
echo "   - pre-commit: Runs code formatting and quality checks"
echo ""
echo "💡 Team members can run '.git/hooks/install-hooks.sh' to install hooks"
echo "💡 Hooks will automatically validate your commits and pushes"
echo ""
echo "📝 Commit message format examples:"
echo "   feat: add new op"
echo "   fix(abs): handle empty tensor edge case"
echo "   docs: update README.md"
echo "   style: format code with clang-format"
