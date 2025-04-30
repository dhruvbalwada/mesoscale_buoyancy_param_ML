import os
import shutil
import yaml
import time

BASE_DIR = "base"
RUNS_DIR = "runs"

def load_config(config_file="experiments.yaml"):
     with open(config_file, "r") as f:
         return yaml.safe_load(f)["experiments"]


def backup_run_dir(src_dir, backup_root, exp_name):
    """
    Copy the existing run directory to a backup location with a timestamp.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_root, f"{exp_name}_{timestamp}")
    shutil.copytree(src_dir, backup_dir)
    print(f"[💾] Backed up {exp_name} to {backup_dir}")



def merge_kv_lines(
    base_text,
    overrides,
    prefix="",
    assign_op="=",
    comment_prefix=None,
    quote_keys=None,
):
    """
    General key=value merger with optional smart quoting.
    
    Parameters:
        base_text       : str (original file contents)
        overrides       : dict (key: value pairs to override or add)
        prefix          : str (text before the key, e.g. '#override ')
        assign_op       : str (e.g. '=', ':')
        comment_prefix  : str or None — restricts which lines to consider
        quote_keys      : set of keys — only these keys will be quoted if values are strings

    Returns:
        str : modified file content
    """

    def format_value(key, val):
        if isinstance(val, str):
            if (quote_keys is None or key in quote_keys):
                if not (val.startswith('"') and val.endswith('"')):
                    return f'"{val}"'
        return val

    lines = base_text.splitlines()
    updated = []
    keys = set(overrides.keys())

    for line in lines:
        stripped = line.strip()
        use_line = True

        if comment_prefix is None or stripped.startswith(comment_prefix):
            if prefix in stripped and assign_op in stripped:
                content = stripped.replace(prefix, "", 1).strip()
                key = content.split(assign_op, 1)[0].strip()

                if key in overrides:
                    val = format_value(key, overrides[key])
                    updated.append(f"{prefix}{key}{assign_op}{val}")
                    keys.remove(key)
                    use_line = False

        if use_line:
            updated.append(line)

    for key in keys:
        val = format_value(key, overrides[key])
        updated.append(f"{prefix}{key}{assign_op}{val}")

    return "\n".join(updated)


def prepare_experiment(
    name,
    config,
    mode="fresh",
    update_submission_script_flag=False,
    update_input_nml_flag=False,
    backup_root=None 
):
    run_dir = os.path.join(RUNS_DIR, name)
    override_path = os.path.join(run_dir, "MOM_override")
    momsub_path = os.path.join(run_dir, "mom.sub")
    nml_path = os.path.join(run_dir, "input.nml")

    # Handle run directory creation/update
    if os.path.exists(run_dir):
        if mode == "skip":
            print(f"[⏭] Skipping existing: {name}")
            return
        elif mode == "fresh":
            shutil.rmtree(run_dir)
            shutil.copytree(BASE_DIR, run_dir)
        elif mode == "update":
            # 🔒 Safety: backup before update
            if backup_root is not None:
                os.makedirs(backup_root, exist_ok=True)
                backup_run_dir(run_dir, backup_root, name)
            # Don't modify files just yet — rest of logic follows
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        shutil.copytree(BASE_DIR, run_dir)

    # === MOM_override ===
    mom_override_cfg = config.get("MOM_override_params", {})
    if os.path.exists(override_path):
        with open(override_path, "r") as f:
            base_override = f.read()
    else:
        base_override = ""
    override_text = merge_kv_lines(base_override, mom_override_cfg, prefix="#override ", assign_op="=", comment_prefix="#override")
    with open(override_path, "w") as f:
        f.write("! Auto-generated MOM_override\n" + override_text)

    # === mom.sub ===
    if update_submission_script_flag:
        mom_sub_cfg = config.get("mom.sub_params", {})
        if os.path.exists(momsub_path):
            with open(momsub_path, "r") as f:
                base_sub = f.read()
        else:
            base_sub = ""
        sub_text = merge_kv_lines(base_sub, mom_sub_cfg, prefix="#SBATCH ", assign_op="=", comment_prefix="#SBATCH", quote_keys={"--job-name"})
        with open(momsub_path, "w") as f:
            f.write(sub_text)

    # === input.nml ===
    if update_input_nml_flag and "input.nml_params" in config:
        nml_cfg = config["input.nml_params"]
        if os.path.exists(nml_path):
            with open(nml_path, "r") as f:
                base_nml = f.read()
        else:
            base_nml = ""
        nml_text = merge_kv_lines(base_nml, nml_cfg, prefix="", assign_op="=", comment_prefix=None)
        with open(nml_path, "w") as f:
            f.write(nml_text)

    print(f"[✓] Prepared ({mode}): {name}")


def generate_all_experiments(mode="fresh", 
                            backup_root=None,
                            update_submission_script_flag=False, 
                            update_input_nml_flag=False,
                            ):
    
    experiments = load_config()
    os.makedirs(RUNS_DIR, exist_ok=True)
    
    for name, config in experiments.items():
        prepare_experiment(
            name,
            config,
            mode=mode,
            update_submission_script_flag=update_submission_script_flag,
            update_input_nml_flag=update_input_nml_flag,
            backup_root = backup_root
        )

if __name__ == "__main__":
    generate_all_experiments(
        mode="update",                      # "fresh", "update", or "skip"
        backup_root="backups",              # optional safety backup
        update_submission_script_flag=True,
        update_input_nml_flag=True
    )