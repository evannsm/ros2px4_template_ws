#!/usr/bin/env python3
# tools/new_pkg_from_template.py
import argparse, re, shutil, stat, sys
from pathlib import Path

TARGET_EXTS = {
    ".py", ".xml", ".cfg", ".ini", ".toml", ".yaml", ".yml",
    ".txt", ".launch.py", ".md", ".rst", ""
}

def log(msg): print(f"[new-pkg] {msg}")

def copy_tree(src: Path, dst: Path, dry=False):
    if not src.exists():
        sys.exit(f"Template not found: {src}")
    if dst.exists():
        sys.exit(f"Destination exists: {dst}")
    log(f"Copy {src} -> {dst}")
    if not dry:
        shutil.copytree(src, dst)

def strip_git_dirs(root: Path, dry=False):
    for p in root.rglob(".git"):
        if p.is_dir():
            log(f"Remove git dir: {p}")
            if not dry:
                shutil.rmtree(p, ignore_errors=True)

def safe_rename(old: Path, new: Path, dry=False):
    if not old.exists(): return
    if new.exists():
        sys.exit(f"Cannot rename {old} -> {new} (dest exists)")
    log(f"Rename {old} -> {new}")
    if not dry:
        old.rename(new)

def replace_in_file(path: Path, replacements, dry=False):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    orig = text
    for pat, repl, is_regex in replacements:
        text = re.sub(pat, repl, text) if is_regex else text.replace(pat, repl)
    if text != orig:
        print(f"[new-pkg] Edit {path}")
        if not dry:
            path.write_text(text, encoding="utf-8")

def make_scripts_executable(pkg_dir: Path, dry=False):
    sd = pkg_dir / "scripts"
    if sd.exists():
        for f in sd.iterdir():
            if f.is_file() and f.suffix == ".py":
                if not dry:
                    f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                log(f"chmod +x {f}")

def build_replacements(old_pkg: str, new_pkg: str, new_node: str):
    # Specific → general; keep raw name replace last
    simple = [
        ("'"+old_pkg+"_node'", f"'{new_node}'"),
        ('"'+old_pkg+'_node"', f'"{new_node}"'),
        (f"{old_pkg} = {old_pkg}", f"{new_pkg} = {new_pkg}"),   # common console-scripts line
        (old_pkg, new_pkg),  # fallback last
    ]
    regex = [
        # package.xml
        (rf"(<name>\s*){re.escape(old_pkg)}(\s*</name>)", r"\1" + new_pkg + r"\2"),
        # setup.cfg / setup.py entry_points
        (rf'(?m)^\s*{re.escape(old_pkg)}\s*=\s*{re.escape(old_pkg)}(\.[\w:]+)', new_pkg + r'\1'),
        (rf'(?m){re.escape(old_pkg)}\s*=\s*{re.escape(old_pkg)}(\.[\w:]+)',
         new_pkg + r'=' + new_pkg + r'\1'),
        # imports
        (rf"\bfrom\s+{re.escape(old_pkg)}\b", "from " + new_pkg),
        (rf"\bimport\s+{re.escape(old_pkg)}\b", "import " + new_pkg),
        # Node name fallback
        (rf"super\(\)\.__init__\(\s*['\"]{re.escape(old_pkg)}_node['\"]\s*\)",
         f"super().__init__('{new_node}')"),
        # pyproject optional: name = "old_pkg"
        (rf'(?m)^(name\s*=\s*")[^"]+(")', r'\1' + new_pkg + r'\2'),
        # [project.scripts] old = "old.main:main" → new = "new.main:main"
        (rf'(?m)^\s*{re.escape(old_pkg)}\s*=\s*"[^\"]*"', f'{new_pkg} = "{new_pkg}.main:main"'),
    ]
    return [(a,b,False) for a,b in simple] + [(a,b,True) for a,b in regex]

def main():
    ap = argparse.ArgumentParser(
        description="Create a new ROS 2 Python package from src/template_pkg (your template).")
    ap.add_argument("package", help="New package name (directory + ROS package name)")
    ap.add_argument("--node-name", help="Node name for super().__init__('...') (default: <package>_node)")
    ap.add_argument("--template", default="src/template_pkg",
                    help="Path to template package (default: src/template_pkg)")
    ap.add_argument("--dest-root", default="src", help="Where to create the new package (default: src)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_pkg = args.package
    new_node = args.node_name or (new_pkg + "_node")
    template_dir = Path(args.template).resolve()
    dest_root = Path(args.dest_root).resolve()
    new_pkg_dir = dest_root / new_pkg

    old_pkg = template_dir.name  # will be 'template_pkg' by default

    copy_tree(template_dir, new_pkg_dir, dry=args.dry_run)
    strip_git_dirs(new_pkg_dir, dry=args.dry_run)

    # Rename inner python package and resource file
    safe_rename(new_pkg_dir / old_pkg, new_pkg_dir / new_pkg, dry=args.dry_run)
    safe_rename(new_pkg_dir / "resource" / old_pkg, new_pkg_dir / "resource" / new_pkg, dry=args.dry_run)

    # Apply replacements
    repls = build_replacements(old_pkg, new_pkg, new_node)
    for p in new_pkg_dir.rglob("*"):
        if p.is_file() and (p.suffix in TARGET_EXTS or p.name.endswith(".launch.py")):
            replace_in_file(p, repls, dry=args.dry_run)

    make_scripts_executable(new_pkg_dir, dry=args.dry_run)
    log("Done." if not args.dry_run else "Dry-run complete.")
    if not args.dry_run:
        print("\nNext:\n  rosdep update && rosdep install --from-paths src --ignore-src -y\n  colcon build\n  source install/setup.bash")

if __name__ == "__main__":
    main()
