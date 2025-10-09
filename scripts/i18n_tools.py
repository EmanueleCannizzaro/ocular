#!/usr/bin/env python3
"""
Lightweight i18n utilities:
- Compile all opendata/locales/*/LC_MESSAGES/messages.po -> messages.mo (no external deps)
- Report translation completeness (total, translated, fuzzy, untranslated)

Usage:
  python scripts/i18n_tools.py --compile --report
"""

from __future__ import annotations

import argparse
import os
import re
import struct
from pathlib import Path
import sys
from typing import Dict, List, Tuple


RE_MSGID = re.compile(r'^msgid\s+"(.*)"$')
RE_MSGSTR = re.compile(r'^msgstr\s+"(.*)"$')
RE_MSGID_PLURAL = re.compile(r'^msgid_plural\s+"(.*)"$')
RE_MSGSTR_N = re.compile(r'^msgstr\[(\d+)\]\s+"(.*)"$')


def _unescape(s: str) -> str:
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\'', "'")


def parse_po(po_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Parse a .po file; return (messages, meta).

    messages: dict of msgid -> [msgstr0, msgstr1, ...]
    meta: header key-values from msgid ''
    """
    messages: Dict[str, List[str]] = {}
    meta: Dict[str, str] = {}

    with po_path.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    idx = 0
    msgid: List[str] | None = None
    msgstrs: Dict[int, List[str]] | None = None
    msgid_plural: List[str] | None = None
    fuzzy = False

    def finalize_entry():
        nonlocal msgid, msgstrs, msgid_plural, fuzzy
        if msgid is None or msgstrs is None:
            return
        key = ''.join(msgid)
        # Header
        if key == '':
            header = ''.join(msgstrs.get(0, []))
            for line in header.split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    meta[k.strip()] = v.strip()
        else:
            max_index = max(msgstrs.keys() or [0])
            arr = [''] * (max_index + 1)
            for i, parts in msgstrs.items():
                arr[i] = ''.join(parts)
            messages[key] = arr
        msgid = None
        msgstrs = None
        msgid_plural = None
        fuzzy = False

    # concatenate continued strings
    def read_quoted(line: str) -> str:
        if line.strip().startswith('"'):
            return _unescape(line.strip().strip('"'))
        return ''

    while idx < len(lines):
        line = lines[idx].rstrip('\n')
        idx += 1

        if line.startswith('#,') and 'fuzzy' in line:
            fuzzy = True
            continue
        if line.startswith('#'):
            continue
        if not line.strip():
            finalize_entry()
            continue

        m = RE_MSGID.match(line)
        if m:
            finalize_entry()
            msgid = [ _unescape(m.group(1)) ]
            msgstrs = {}
            msgid_plural = None
            # collect continued msgid lines
            while idx < len(lines) and lines[idx].lstrip().startswith('"'):
                msgid.append(read_quoted(lines[idx]))
                idx += 1
            continue

        m = RE_MSGID_PLURAL.match(line)
        if m and msgid is not None:
            msgid_plural = [ _unescape(m.group(1)) ]
            while idx < len(lines) and lines[idx].lstrip().startswith('"'):
                msgid_plural.append(read_quoted(lines[idx]))
                idx += 1
            continue

        m = RE_MSGSTR_N.match(line)
        if m and msgid is not None:
            n = int(m.group(1))
            msgstrs.setdefault(n, []).append(_unescape(m.group(2)))
            while idx < len(lines) and lines[idx].lstrip().startswith('"'):
                msgstrs[n].append(read_quoted(lines[idx]))
                idx += 1
            continue

        m = RE_MSGSTR.match(line)
        if m and msgid is not None:
            msgstrs.setdefault(0, []).append(_unescape(m.group(1)))
            while idx < len(lines) and lines[idx].lstrip().startswith('"'):
                msgstrs[0].append(read_quoted(lines[idx]))
                idx += 1
            continue

    finalize_entry()
    return messages, meta


def write_mo(messages: Dict[str, List[str]], mo_path: Path, meta: Dict[str, str] | None = None) -> None:
    # Build catalog: map from original string to translated string (singular form only)
    catalog: Dict[str, str] = {k: v[0] if v else '' for k, v in messages.items()}

    # Ensure header exists with UTF-8 charset so gettext decodes properly
    header_lines = [
        'Project-Id-Version: PACKAGE VERSION',
        'Report-Msgid-Bugs-To: ',
        'POT-Creation-Date: ',
        'PO-Revision-Date: ',
        'Last-Translator: ',
        'Language-Team: ',
        'MIME-Version: 1.0',
        'Content-Type: text/plain; charset=UTF-8',
        'Content-Transfer-Encoding: 8bit',
    ]
    if meta:
        for k, v in meta.items():
            if k.lower().startswith('content-type'):
                continue
            header_lines.append(f"{k}: {v}")
    header = '\n'.join(header_lines) + '\n'
    catalog.setdefault('', header)

    # The .mo format:
    # magic, revision, nstrings, orig_tab_offset, trans_tab_offset, hash_size, hash_offset
    offsets = []
    ids = list(catalog.keys())
    strs = [catalog[i] for i in ids]

    # Sort by original ids
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids = [ids[i] for i in order]
    strs = [strs[i] for i in order]

    keystart = 7 * 4 + 16 * len(ids)
    valuestart = keystart + sum(len(s.encode('utf-8')) + 1 for s in ids)

    koffsets = []
    toffsets = []
    kpos = keystart
    vpos = valuestart

    for s in ids:
        s_bytes = s.encode('utf-8') + b'\0'
        koffsets.append((len(s_bytes) - 1, kpos))
        kpos += len(s_bytes)
    for s in strs:
        s_bytes = s.encode('utf-8') + b'\0'
        toffsets.append((len(s_bytes) - 1, vpos))
        vpos += len(s_bytes)

    # Header
    output = []
    output.append(struct.pack("Iiiiiii", 0x950412de, 0, len(ids), 7 * 4, 7 * 4 + len(ids) * 8, 0, 0))

    # Index tables
    for length, offset in koffsets:
        output.append(struct.pack("ii", length, offset))
    for length, offset in toffsets:
        output.append(struct.pack("ii", length, offset))

    # Original strings
    for s in ids:
        output.append(s.encode('utf-8') + b"\0")
    # Translated strings
    for s in strs:
        output.append(s.encode('utf-8') + b"\0")

    mo_path.parent.mkdir(parents=True, exist_ok=True)
    with mo_path.open('wb') as f:
        for chunk in output:
            f.write(chunk)


def compile_po_to_mo(po_path: Path, mo_path: Path) -> None:
    messages, meta = parse_po(po_path)
    write_mo(messages, mo_path, meta)


def compute_stats(po_path: Path) -> Dict[str, int]:
    with po_path.open('r', encoding='utf-8') as f:
        content = f.read()
    # Split entries by blank lines, naive but effective for stats
    entries = [e for e in re.split(r"\n\s*\n", content) if 'msgid' in e]
    total = 0
    translated = 0
    fuzzy = 0
    for e in entries:
        # skip header where msgid ""
        if re.search(r'^msgid\s+""', e, re.M):
            continue
        total += 1
        if ', fuzzy' in e:
            fuzzy += 1
        # consider translated if any msgstr has non-empty content
        has_translation = bool(re.search(r'^msgstr(?:\[\d+\])?\s+".+"', e, re.M)) and not re.search(r'^msgstr(?:\[\d+\])?\s+""\s*$', e, re.M)
        if has_translation:
            translated += 1
    return {
        'total': total,
        'translated': translated,
        'fuzzy': fuzzy,
        'untranslated': max(total - translated, 0),
    }


def find_locales(base: Path) -> List[str]:
    locales = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        po = d / 'LC_MESSAGES' / 'messages.po'
        if po.exists():
            locales.append(d.name)
    return sorted(locales)


def main():
    parser = argparse.ArgumentParser(description='Compile and report i18n catalogs')
    parser.add_argument('--compile', action='store_true', help='Compile .po to .mo for all locales')
    parser.add_argument('--report', action='store_true', help='Print completeness report')
    parser.add_argument('--extract-pot', action='store_true', help='Generate messages.pot from source locale (en_US)')
    parser.add_argument('--merge', action='store_true', help='Merge messages.pot into all locale .po (add missing entries)')
    parser.add_argument('--checklist', action='store_true', help='Write UNTRANSLATED.txt per locale with missing strings')
    parser.add_argument('--base', default='opendata/locales', help='Locales base directory')
    parser.add_argument('--require-complete', nargs='*', default=[], help='Locales that must have zero untranslated; non-zero exit if not met')
    args = parser.parse_args()

    base = Path(args.base)
    locales = find_locales(base)
    if not locales:
        print(f'No locales found in {base}')
        return

    # Optional: extract .pot from en_US
    if args.extract_pot:
        src_po = base / 'en_US' / 'LC_MESSAGES' / 'messages.po'
        pot = base / 'messages.pot'
        try:
            msgs, meta = parse_po(src_po)
            _write_pot(msgs, pot, meta)
            print(f'[OK] Extracted template {pot} from {src_po}')
        except Exception as e:
            print(f'[ERR] Failed to extract POT: {e}')

    # Optional: merge missing entries into each locale
    if args.merge:
        pot = base / 'messages.pot'
        if not pot.exists():
            print(f'[WARN] {pot} not found; generating from en_US')
            src_po = base / 'en_US' / 'LC_MESSAGES' / 'messages.po'
            msgs, meta = parse_po(src_po)
            _write_pot(msgs, pot, meta)
        pot_msgs, _ = parse_po(pot)
        pot_ids = set(pot_msgs.keys())
        for loc in locales:
            po = base / loc / 'LC_MESSAGES' / 'messages.po'
            if loc == 'en_US':
                continue
            try:
                loc_msgs, _ = parse_po(po)
                missing = sorted(pot_ids - set(loc_msgs.keys()))
                if missing:
                    _append_missing(po, missing)
                    print(f'[OK] {loc}: appended {len(missing)} new entries')
                else:
                    print(f'[OK] {loc}: up-to-date')
            except Exception as e:
                print(f'[ERR] {loc}: merge failed: {e}')

    if args.compile:
        for loc in locales:
            po = base / loc / 'LC_MESSAGES' / 'messages.po'
            mo = base / loc / 'LC_MESSAGES' / 'messages.mo'
            try:
                compile_po_to_mo(po, mo)
                print(f'[OK] Compiled {po} -> {mo}')
            except Exception as e:
                print(f'[ERR] Failed {po}: {e}')

    failed = False
    if args.report or args.checklist or args.require_complete:
        print('\nTranslation completeness:')
        for loc in locales:
            po = base / loc / 'LC_MESSAGES' / 'messages.po'
            try:
                s = compute_stats(po)
                pct = 0 if s['total'] == 0 else int(round(s['translated'] * 100.0 / s['total']))
                print(f"- {loc}: {pct}% translated (total={s['total']}, translated={s['translated']}, fuzzy={s['fuzzy']}, untranslated={s['untranslated']})")
                if args.checklist and s['untranslated'] > 0:
                    _write_untranslated_list(po, base / loc / 'UNTRANSLATED.txt')
                    print(f"  -> Wrote {loc}/UNTRANSLATED.txt")
                if loc in set(args.require_complete) and s['untranslated'] > 0:
                    failed = True
            except Exception as e:
                print(f'- {loc}: error computing stats: {e}')
                if loc in set(args.require_complete):
                    failed = True

    if failed:
        print('\nERROR: Some required locales are incomplete.')
        sys.exit(2)


# ----------------- Helpers for POT/merge/checklist -----------------

def _escape_po(s: str) -> str:
    return s.replace('\\', r'\\').replace('"', r'\"').replace('\n', r'\n')


def _write_pot(messages: Dict[str, List[str]], pot_path: Path, meta: Dict[str, str] | None = None) -> None:
    pot_path.parent.mkdir(parents=True, exist_ok=True)
    with pot_path.open('w', encoding='utf-8') as f:
        # Basic header
        f.write('# Generated POT template\n')
        f.write('msgid ""\n')
        f.write('msgstr ""\n')
        f.write('"Content-Type: text/plain; charset=UTF-8\\n"\n')
        f.write('"Content-Transfer-Encoding: 8bit\\n"\n\n')
        for msgid in sorted(messages.keys()):
            if msgid == '':
                continue
            f.write(f'msgid "{_escape_po(msgid)}"\n')
            f.write('msgstr ""\n\n')


def _append_missing(po_path: Path, missing_msgids: List[str]) -> None:
    with po_path.open('a', encoding='utf-8') as f:
        for msgid in missing_msgids:
            f.write('\n')
            f.write(f'msgid "{_escape_po(msgid)}"\n')
            f.write('msgstr ""\n')


def _write_untranslated_list(po_path: Path, out_path: Path) -> None:
    # list msgids with empty msgstr
    messages, _ = parse_po(po_path)
    untranslated = [mid for mid, strs in messages.items() if mid and (not strs or all(s.strip() == '' for s in strs))]
    out_path.write_text('\n'.join(sorted(untranslated)), encoding='utf-8')


if __name__ == '__main__':
    main()
