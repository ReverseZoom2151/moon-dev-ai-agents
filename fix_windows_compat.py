"""
Script to fix Windows compatibility issues in all agent files
- Replaces hardcoded macOS paths with cross-platform PROJECT_ROOT paths
- Adds UTF-8 console encoding
- Ensures proper sys.path setup
"""

import re
import sys
from pathlib import Path

# Fix Windows console encoding for emojis (for this script itself)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# Get project root
PROJECT_ROOT = Path(__file__).parent

# Files to fix
agent_files = [
    "src/agents/chat_agent_ad.py",
    "src/agents/copybot_agent.py",
    "src/agents/focus_agent.py",
    "src/agents/rbi_agent.py",
    "src/agents/rbi_agent_v2.py",
    "src/agents/rbi_agent_v3.py",
    "src/agents/rbi_agent_v2_simple.py",
    "src/agents/rbi_batch_backtester.py",
    "src/agents/shortvid_agent.py",
    "src/agents/sniper_agent.py",
    "src/agents/tweet_agent.py",
    "src/agents/tx_agent.py",
]

def fix_file(file_path):
    """Fix Windows compatibility in a single file"""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"{'='*60}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # 1. Replace hardcoded macOS paths with PROJECT_ROOT-based paths
    replacements = [
        # moon-dev-ai-agents-for-trading paths
        (r'"/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/([^"]+)"',
         r'PROJECT_ROOT / "src" / "data" / "\1"'),
        (r"'/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/([^']+)'",
         r'PROJECT_ROOT / "src" / "data" / "\1"'),
        (r'"/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading([^"]+)"',
         r'PROJECT_ROOT / "\1"'),
        (r"'/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading([^']+)'",
         r'PROJECT_ROOT / "\1"'),

        # solana-copy-trader paths
        (r'"/Users/md/Dropbox/dev/github/solana-copy-trader/([^"]+)"',
         r'PROJECT_ROOT.parent / "solana-copy-trader" / "\1"'),
        (r"'/Users/md/Dropbox/dev/github/solana-copy-trader/([^']+)'",
         r'PROJECT_ROOT.parent / "solana-copy-trader" / "\1"'),

        # Untitled/sounds paths (generic sounds directory)
        (r'"/Users/md/Dropbox/dev/github/Untitled/sounds/([^"]+)"',
         r'PROJECT_ROOT / "sounds" / "\1"'),
        (r"'/Users/md/Dropbox/dev/github/Untitled/sounds/([^']+)'",
         r'PROJECT_ROOT / "sounds" / "\1"'),
    ]

    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes.append(f"  ✅ Replaced hardcoded path pattern: {pattern[:50]}...")

    # 2. Fix Path() construction to use proper / operator
    # Convert Path("string") to use / operators when combined with PROJECT_ROOT
    def fix_path_construction(match):
        path_str = match.group(1)
        # Split by / and create proper Path construction
        parts = [p.strip('"\'') for p in path_str.split('/') if p.strip('"\'')]
        return f'PROJECT_ROOT / "' + '" / "'.join(parts) + '"'

    # 3. Ensure PROJECT_ROOT is defined early
    if 'PROJECT_ROOT' not in content:
        # Find where to insert PROJECT_ROOT definition
        import_section_end = content.find('\n\n', content.find('import'))
        if import_section_end > 0:
            insertion_point = import_section_end + 2
            project_root_def = """# Get project root directory
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
            content = content[:insertion_point] + project_root_def + content[insertion_point:]
            changes.append("  ✅ Added PROJECT_ROOT definition")

    # 4. Add UTF-8 encoding if not present
    if 'sys.stdout.reconfigure' not in content and 'import sys' in content:
        # Find import sys
        sys_import_pos = content.find('import sys')
        if sys_import_pos > 0:
            # Find end of imports section
            next_section = content.find('\n\n', sys_import_pos)
            if next_section > 0:
                utf8_code = """\n\n# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass"""
                content = content[:next_section] + utf8_code + content[next_section:]
                changes.append("  ✅ Added UTF-8 console encoding")

    # 5. Fix sys.path.append to use PROJECT_ROOT
    content = re.sub(
        r"sys\.path\.append\(['\"]?/Users/md/[^)]+\)",
        "# sys.path already set up with PROJECT_ROOT above",
        content
    )
    if re.search(r"sys\.path\.append\(['\"]?/Users", original_content):
        changes.append("  ✅ Removed hardcoded sys.path.append")

    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print("\n📝 Changes made:")
        for change in changes:
            print(change)
        print(f"\n✅ Successfully updated: {file_path}")
        return True
    else:
        print("ℹ️  No changes needed")
        return False

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Windows Compatibility Fix Script")
    print(f"📂 Project Root: {PROJECT_ROOT}")

    fixed_count = 0
    for file_path in agent_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            if fix_file(full_path):
                fixed_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")

    print(f"\n{'='*60}")
    print(f"🎉 Completed! Fixed {fixed_count}/{len(agent_files)} files")
    print(f"{'='*60}")
