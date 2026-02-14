#!/bin/bash
# Team-wide hook installation script
# This script installs all Git hooks from the hooks/ directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we're in the right directory
if [ ! -d "$SCRIPT_DIR/hooks" ]; then
    echo "❌ Error: hooks directory not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

echo "🚀 Setting up team Git hooks..."
echo ""

# Run the hooks installer
"$SCRIPT_DIR/hooks/install-hooks.sh"

echo ""
echo "✅ Hook setup complete!"
echo ""
echo "👥 For team members:"
echo "   - Run './install-hooks.sh' to install all hooks"
echo "   - Hooks will automatically validate commits for code quality and message format"
echo "   - Commit messages must follow conventional commits format"
echo ""
echo "📚 Need help? Check the README.md for more details."