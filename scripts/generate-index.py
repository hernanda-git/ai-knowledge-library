#!/usr/bin/env python3
"""Generate INDEX.md — a flat, auto-generated document index for the AI Knowledge Library."""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_DIRS = {'_Meta', 'ai-library', '.git', '.venv', '.lib', 'snapshots',
                'node_modules', '.github', 'removed'}
SKIP_FILES = {'README.md', 'CATALOG.md', 'INDEX.md', 'LICENSE', 'CONTRIBUTING.md', 'SKILL.md'}

def get_title(filepath):
    """Extract the first H1 from a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# ') and not line.startswith('#  '):
                    return line.lstrip('# ').strip()
                if line.startswith('---'):
                    break  # frontmatter
        return None
    except Exception:
        return None

def collect_docs(root):
    """Walk the repo and collect all markdown files by category."""
    categories = {}

    for entry in sorted(os.listdir(root)):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in EXCLUDE_DIRS:
            continue
        # Check if it's a numbered category directory
        if not re.match(r'^\d{2}-', entry):
            continue

        docs = []
        for fname in sorted(os.listdir(entry_path)):
            if not fname.endswith('.md'):
                continue
            if fname in SKIP_FILES:
                continue
            fpath = os.path.join(entry_path, fname)
            title = get_title(fpath) or fname
            docs.append((fname, title))

        # Check for nested sub-topics
        sub_topics = {}
        for fname in sorted(os.listdir(entry_path)):
            sub_path = os.path.join(entry_path, fname)
            if not os.path.isdir(sub_path):
                continue
            sub_docs = []
            for sf in sorted(os.listdir(sub_path)):
                if not sf.endswith('.md'):
                    continue
                sfpath = os.path.join(sub_path, sf)
                stitle = get_title(sfpath) or sf
                sub_docs.append((sf, stitle))
            if sub_docs:
                sub_topics[fname] = sub_docs

        if docs or sub_topics:
            categories[entry] = (docs, sub_topics)

    return categories

def generate(categories):
    """Produce INDEX.md content."""
    lines = []
    lines.append('# 📚 AI Knowledge Library — Complete Document Index')
    lines.append('')
    lines.append('> Auto-generated index. Last updated: %s' % 
                 __import__('datetime').datetime.now().strftime('%Y-%m-%d'))
    lines.append('')
    lines.append('| %s | %s | %s |' % ('#', 'Category', 'Documents'))
    lines.append('| --- | -------- | --------- |')
    
    all_files = 0
    for cat in sorted(categories.keys(), key=lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else 999):
        docs, sub_topics = categories[cat]
        cat_name = cat[3:].replace('-', ' ').title()
        doc_list = ', '.join(d[1] for d in docs)
        all_files += len(docs)
        
        # Main category line
        short = cat.split('-')[0]
        lines.append('| `%s` | **%s** | %s |' % (short, cat_name, doc_list if doc_list else '*sub-topics only*'))
        
        # Sub-topics
        for sub_cat, sub_docs in sorted(sub_topics.items()):
            sub_name = sub_cat[3:].replace('-', ' ').title()
            sub_list = ', '.join(sd[1] for sd in sub_docs)
            lines.append('| `→` | &nbsp;&nbsp;_%s_ | %s |' % (sub_name, sub_list))
            all_files += len(sub_docs)
    
    lines.append('')
    lines.append('**Total documents indexed: %d**' % all_files)
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*See [CATALOG.md](CATALOG.md) for the structural hierarchy and merge map.*')
    lines.append('')
    return '\n'.join(lines)

if __name__ == '__main__':
    categories = collect_docs(REPO_ROOT)
    content = generate(categories)
    out_path = os.path.join(REPO_ROOT, 'INDEX.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ INDEX.md generated — %d categories, %d documents indexed' % (
        len(categories), content.count('- `') - 1))  # approximate
