# ROS 2 / PX4 Template Workspace

## ✨ Features

- Submodules for deps (`px4_msgs`, `mocap4r2_msgs`, `ROS2Logger`, `test_logger`)
- Reusable **template package** at `src/template_pkg/` (kept out of builds)
- One-liner generator: `ros2px4 pkg create --name <PACKAGE> [--node-name <NODE>]`
- Auto-rewrites `package.xml`, imports, console scripts, and `super().__init__('…')`

---

## 🚀 Quick Start

1. Make a workspace directory

```bash
mkdir my_new_ws
```

2. Clone this into the workspace

```bash
git clone git@github.com:evannsm/ros2px4_template_ws.git --recursive
```

3. Add CLI tools to path

```bash
chmod +x tools/ros2px4
mkdir -p ~/.local/bin
ln -sf "$(pwd)/tools/ros2px4" ~/.local/bin/ros2px4

# Add to PATH for future shells
# Bash:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Reload shell so PATH takes effect now
exec $SHELL -l

# Verify
which ros2px4
ros2px4 --help
```

4. Create a new package from the template

```bash
# From repo root
ros2px4 pkg create --name my_new_pkg

# Customize node name (default: <name>_node)
ros2px4 pkg create --name my_new_pkg --node-name my_node

# Overwrite an existing target directory if needed
ros2px4 pkg create --name my_new_pkg --force
```

`````

# 🛠️ What the generator does

- **Copies** `src/template_pkg/` → `src/<name>/`
- **Renames**
  - Python package dir: `template_pkg/` → `<name>/`
  - Ament resource: `resource/template_pkg` → `resource/<name>`
- **Rewrites**
  - `package.xml` `<name>`
  - `setup.cfg` / `setup.py` / `pyproject.toml` console scripts
  - Imports: `from template_pkg …` → `from <name> …`
  - `super().__init__('template_pkg_node')` → `--node-name` (default `<name>_node`)
- **Marks** `scripts/*.py` executable

### 🧭 Repository Layout (after setup)

````text
ros2px4_template_ws/
├── src/
│   ├── px4_msgs         # submodule (e.g., release/1.15)
│   ├── mocap4r2_msgs    # submodule
│   ├── ROS2Logger       # submodule
│   ├── test_logger      # submodule
│   ├── template_pkg     # the template (has COLCON_IGNORE)
│   └── my_new_pkg       # generated from template
└── tools/
    ├── ros2px4          # CLI (subcommand: pkg create)
    └── new_pkg_from_template.py


### 🛠️ Troubleshooting

- **`ros2px4: command not found`**
  Use `./tools/ros2px4 …` or add it to PATH (see Step 3B).

- **`Permission denied` when running `ros2px4`**
  ```bash
  chmod +x tools/ros2px4
`````

- **`Permission denied` when running `ros2px4`**

  ```bash
  ros2px4 pkg create --name <pkg> --force
  # or
  rm -rf src/<pkg>
  ```

- **`Verify rewrites completed`**
  ```bash
  grep -R "template_pkg" -n src/my_new_pkg || echo "✅ no template_pkg refs"
  grep -R "super().__init__" -n src/my_new_pkg
  ```
