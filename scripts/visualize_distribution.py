
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
from scipy.stats import gaussian_kde
from Bio.PDB import PDBParser
from tqdm import tqdm
import warnings

# Suppress PDB warnings
warnings.simplefilter('ignore', category=UserWarning)

# Import local modules
import sys
if "." not in sys.path:
    sys.path.append(".")

from rna_backbone_design.analysis.ensemble_metrics import _pairwise_rmsd_matrix

def get_c4_coords(pdb_file):
    """
    Parses a PDB file and returns the coordinates of C4' atoms (or P if C4' missing).
    Returns: numpy array of shape (L, 3)
    """
    parser = PDBParser()
    try:
        structure = parser.get_structure('struct', pdb_file)
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None

    coords = []
    # Assume single model
    try:
        model = next(iter(structure))
        for chain in model:
            for residue in chain:
                # Skip heteroatoms
                if residue.id[0] != ' ':
                    continue
                
                atom = None
                # Prioritize C4' as it's a key backbone atom for RNA
                if "C4'" in residue:
                    atom = residue["C4'"]
                elif "P" in residue:
                    atom = residue["P"]
                
                if atom:
                    coords.append(atom.get_coord())
    except StopIteration:
        pass
        
    return np.array(coords)

def main():
    parser = argparse.ArgumentParser(description="Visualize RNA structure distribution via MDS on RMSD matrix.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing generated PDB files.")
    parser.add_argument("--gt_pdb", type=str, default=None, help="Path to ground truth PDB file (optional).")
    parser.add_argument("--output", type=str, default="distribution_viz.png", help="Output plot filename.")
    parser.add_argument("--title", type=str, default="Ensemble Distribution", help="Plot title.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    pdb_files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.pdb')])
    if not pdb_files:
        print(f"No PDB files found in {args.input_dir}")
        return

    print(f"Found {len(pdb_files)} generated structures.")

    all_coords = []
    
    # Load GT if exists
    gt_coords_entry = None
    if args.gt_pdb:
        if os.path.exists(args.gt_pdb):
            print(f"Loading GT: {args.gt_pdb}")
            gt_c = get_c4_coords(args.gt_pdb)
            if gt_c is not None and len(gt_c) > 0:
                gt_coords_entry = gt_c
            else:
                print("Failed to load GT coords.")
        else:
            print(f"Warning: GT PDB {args.gt_pdb} not found.")

    # Load Generated
    print("Loading generated structures...")
    temp_coords = []
    
    for f in tqdm(pdb_files):
        c = get_c4_coords(f)
        if c is not None and len(c) > 0:
            temp_coords.append(c)
    
    if len(temp_coords) < 2:
        print("Need at least 2 structures to plot distribution.")
        return

    # Check shapes
    lengths = [len(c) for c in temp_coords]
    from collections import Counter
    common_len = Counter(lengths).most_common(1)[0][0]
    
    # Filter generated
    for c in temp_coords:
        if len(c) == common_len:
            all_coords.append(c)
            
    # Add GT if it matches length
    gt_idx = -1
    if gt_coords_entry is not None:
        if len(gt_coords_entry) == common_len:
            all_coords.append(gt_coords_entry)
            gt_idx = len(all_coords) - 1
        else:
             print(f"Warning: GT length ({len(gt_coords_entry)}) mismatches common length ({common_len}). GT will not be included in MDS.")
    
    print(f"Using {len(all_coords)} structures with length {common_len}.")

    # Convert to tensor for RMSD calculation
    coords_tensor = torch.tensor(np.array(all_coords), dtype=torch.float32) # (N, L, 3)
    L = coords_tensor.shape[1]
    mask = torch.ones(L, dtype=torch.float32)

    print("Calculating pairwise RMSD matrix...")
    if torch.cuda.is_available():
        coords_tensor = coords_tensor.cuda()
        mask = mask.cuda()
    
    rmsd_matrix = _pairwise_rmsd_matrix(coords_tensor, mask)
    rmsd_matrix_np = rmsd_matrix.cpu().numpy()

    print("Running MDS...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=-1, max_iter=300)
    coords_2d = mds.fit_transform(rmsd_matrix_np)

    # Separate generated and GT 2D coords
    if gt_idx != -1:
        gen_coords_2d = coords_2d[:-1]
        gt_coord_2d = coords_2d[-1]
    else:
        gen_coords_2d = coords_2d
        gt_coord_2d = None

    # --- Plotting ---
    print("Plotting...")
    
    # Nature-style aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Custom Colormap (Light to Dark for Density)
    # White/Light Blue -> Dark Blue
    colors = ["#FFFFFF", "#3040A0"] 
    cmap_name = "custom_rna_density"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Layout: 
    # Main Scatter (Center)
    # Top Marginal (Top)
    # Right Marginal (Right)
    
    fig = plt.figure(figsize=(8, 8))
    
    # Define grid
    # Width ratios: Main(4), RightMarginal(1)
    # Height ratios: TopMarginal(1), Main(4)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                           wspace=0.05, hspace=0.05)

    # Axes
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # KDE Calculation
    x = gen_coords_2d[:, 0]
    y = gen_coords_2d[:, 1]
    
    # Calculate robust limits
    q1_x, q3_x = np.percentile(x, [5, 95])
    iqr_x = q3_x - q1_x
    view_min_x = x.min() - 0.1 * (x.max() - x.min())
    view_max_x = x.max() + 0.1 * (x.max() - x.min())
    
    # Focus closer if outliers exist
    if iqr_x > 0:
        view_min_x = max(view_min_x, q1_x - 1.5 * iqr_x)
        view_max_x = min(view_max_x, q3_x + 1.5 * iqr_x)
    
    q1_y, q3_y = np.percentile(y, [5, 95])
    iqr_y = q3_y - q1_y
    view_min_y = y.min() - 0.1 * (y.max() - y.min())
    view_max_y = y.max() + 0.1 * (y.max() - y.min())

    if iqr_y > 0:
        view_min_y = max(view_min_y, q1_y - 1.5 * iqr_y)
        view_max_y = min(view_max_y, q3_y + 1.5 * iqr_y)

    # Grid for KDE
    xi, yi = np.mgrid[view_min_x:view_max_x:100j, view_min_y:view_max_y:100j]
    positions = np.vstack([xi.flatten(), yi.flatten()])
    values = np.vstack([x, y])
    
    try:
        k = gaussian_kde(values)
        zi = k(positions)
        zi = zi.reshape(xi.shape)
        
        # 1. Main Plot: Filled Contours
        # Using the custom colormap with white background (transparent)
        # Use more levels for smooth gradient
        # First layer: white background is default for matplotlib figure
        # Contourf with alpha will blend with white
        
        # Smoother distribution requested
        k.set_bandwidth(bw_method=k.factor) 
        zi = k(positions)
        zi = zi.reshape(xi.shape)

        # Plot contourf
        # We start from a low threshold to ignore very low density areas (make them white)
        # levels = np.linspace(zi.min(), zi.max(), 20)
        # Or better: quantile based levels
        
        ax_main.contourf(xi, yi, zi, levels=20, cmap=cm, alpha=0.9, extend='both')
        
        # Optional: Add very subtle contour lines for definition
        ax_main.contour(xi, yi, zi, levels=20, colors='white', linewidths=0.5, alpha=0.3)
        
    except Exception as e:
        print(f"KDE failed: {e}")
        # Fallback scatter if KDE fails (should be rare)
        ax_main.scatter(x, y, c="#3040A0", alpha=0.5)

    # Plot GT
    if gt_coord_2d is not None:
        ax_main.scatter(gt_coord_2d[0], gt_coord_2d[1], 
                       c='red', marker='*', s=200, edgecolors='white', linewidth=1.5, zorder=10, label='Ground Truth')
        ax_main.legend(loc='upper right', frameon=False, fontsize=10)

    # Clean Main Axes
    ax_main.set_xlim(view_min_x, view_max_x)
    ax_main.set_ylim(view_min_y, view_max_y)
    # ax_main.set_xticks([])
    # ax_main.set_yticks([])
    # ax_main.set_xlabel("MDS Dimension 1")
    # ax_main.set_ylabel("MDS Dimension 2")
    
    # Remove top and right spines for Nature style
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.tick_params(direction='out')

    # 2. Top Marginal
    # Use the primary color from the palette (#3040A0)
    marginal_color = "#3040A0"
    
    # Plot histogram (gray bars) AND density line (colored)
    ax_top.hist(x, bins=30, density=True, color='gray', alpha=0.3, edgecolor='white')
    
    try:
        x_grid = np.linspace(view_min_x, view_max_x, 200)
        kde_x = gaussian_kde(x)
        kde_x.set_bandwidth(bw_method=kde_x.factor * 1.5)
        y_vals = kde_x(x_grid)
        ax_top.plot(x_grid, y_vals, color=marginal_color, linewidth=1.5)
        # ax_top.fill_between(x_grid, y_vals, color=marginal_color, alpha=0.1) # Keep white background mostly
    except:
        pass
        
    ax_top.axis('off')

    # 3. Right Marginal
    # Plot histogram (gray bars) AND density line (colored)
    ax_right.hist(y, bins=30, density=True, orientation='horizontal', color='gray', alpha=0.3, edgecolor='white')
    
    try:
        y_grid = np.linspace(view_min_y, view_max_y, 200)
        kde_y = gaussian_kde(y)
        kde_y.set_bandwidth(bw_method=kde_y.factor * 1.5)
        x_vals = kde_y(y_grid)
        ax_right.plot(x_vals, y_grid, color=marginal_color, linewidth=1.5)
        # ax_right.fill_betweenx(y_grid, x_vals, x2=0, color=marginal_color, alpha=0.1)
    except:
        pass
        
    ax_right.axis('off')

    # Add Title
    # plt.suptitle(args.title, fontsize=14, y=0.95)
    
    # Save
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main()
