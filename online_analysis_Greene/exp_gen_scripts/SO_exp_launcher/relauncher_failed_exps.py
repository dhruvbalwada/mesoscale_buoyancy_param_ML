import os
import subprocess
from pathlib import Path

def check_job_failed(experiment_folder):
    """Check if experiment failed - if .nc files exist, it failed and needs rerun."""
    folder = Path(experiment_folder)
    if not folder.exists():
        return True  # Folder doesn't exist, consider it failed
    
    nc_files = list(folder.glob('*.nc'))
    return len(nc_files) > 0  # If .nc files exist, job failed

def rerun_incomplete_experiments(base_folder, res):
    """Rerun failed experiments for a given resolution."""
    failed = []
    
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
        for cb in [0., 1., 2., 3., 4.]:
            for cu in [0., 0.5, 1., 2.]:
                exp_folder = f'{base_folder}/tau_{tau}_cb_{cb}_cu_{cu}'
                
                if check_job_failed(exp_folder):
                    failed.append((exp_folder, res, tau, cb, cu))
    
    print(f"\nFound {len(failed)} failed experiments for {res}")
    
    if len(failed) == 0:
        return
    
    print("\nFailed experiments:")
    for exp_folder, _, tau, cb, cu in failed:
        print(f"  tau={tau}, cb={cb}, cu={cu}")
    
    # Ask for confirmation before rerunning
    confirm = input(f"\nRerun {len(failed)} experiments? (y/n): ")
    if confirm.lower() != 'y':
        return
    
    for exp_folder, _, tau, cb, cu in failed:
        print(f"\nRerunning: tau={tau}, cb={cb}, cu={cu}")
        
        folder_path = Path(exp_folder)
        
        # Check if folder and mom.sub exist
        if not folder_path.exists():
            print(f"  ✗ Folder not found: {exp_folder}")
            continue
            
        mom_sub = folder_path / 'mom.sub'
        if not mom_sub.exists():
            print(f"  ✗ mom.sub not found")
            continue
        
        # Delete .nc files
        nc_files = list(folder_path.glob('*.nc'))
        for nc_file in nc_files:
            print(f"  Removing {nc_file.name}")
            nc_file.unlink()
        
        try:
            # Submit job from experiment folder
            result = subprocess.run(['sbatch', 'mom.sub'], cwd=exp_folder, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ Resubmitted: {result.stdout.strip()}")
            else:
                print(f"  ✗ Error: {result.stderr.strip()}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == '__main__':
    base_path = '/scratch/db194/mom6/feb2026/channel_extra_sponge_slow_woc_'
    
    for res in ['p25', 'p5', '1p0']:
        print(f"\n{'='*60}")
        print(f"Checking {res}")
        print(f"{'='*60}")
        
        base_folder = base_path + res
        rerun_incomplete_experiments(base_folder, res)