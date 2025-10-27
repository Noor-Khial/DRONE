# TWTD sensing: probability model with CFAR coupling
# Toggles mapped to reviewer asks:
#   [R8-53, R8-81] TWTD vs LOS ablation (use_twtd)
#   [R8-54]       Processing gain sweep (G_proc)
#   [R6-4, R8-36] Adversarial levels (attenuation)

import math
import numpy as np
import sys

# Try local import first; fall back to repo-style path if needed.
try:
    from cacfar import get_cacfar_detection
except Exception:
    sys.path.append('.')
    from throughWallDetection.cacfar import get_cacfar_detection  # type: ignore

# Optional helpers; safe fallbacks if env deps aren’t present.
try:
    from env.targets_movements.load_obstacles import load_obstacles
    from env.targets_movements.utils import is_point_in_obstacle
except Exception:
    def load_obstacles(_): return []
    def is_point_in_obstacle(x, y, obstacles): return False

# ------------------------------
# Global experiment toggles
# ------------------------------
_USE_TWTD = True      # [R8-53, R8-81]
_G_PROC  = 1.0        # [R8-54]
_ADV_LVL = 0.0        # [R6-4, R8-36] 0.0 (none) .. 1.0 (strong attenuation)

def set_twtd_mode(enabled: bool):
    """Enable/disable TWTD (False -> LOS baseline).  [R8-53, R8-81]"""
    global _USE_TWTD
    _USE_TWTD = bool(enabled)

def set_twtd_gain(gain: float):
    """Set processing gain multiplier for CFAR peak strength.  [R8-54]"""
    global _G_PROC
    _G_PROC = float(gain)

def set_adversarial_level(level: float):
    """Set detection attenuation level in [0,1].  [R6-4, R8-36]"""
    global _ADV_LVL
    _ADV_LVL = float(max(0.0, min(1.0, level)))

def calculate_detection_probability(
    d,             # distance to target (meters)
    s_proc=15,     # processed SNR
    s_th=10,       # SNR threshold
    phi_proc=5,    # placeholder
    phi_base=3,    # placeholder
    m=0.5,         # material attenuation factor
    f_v=20,        # vibration frequency (Hz)
    kappa_snr=0.5, # SNR constant
    beta=0.2,      # material constant
    gamma=0.001,   # placeholder
    obstacle=False,
    obstacle_attenuation=0.4,
    use_twtd=None,   # override global if not None  [R8-53, R8-81]
    G_proc=None      # override global if not None  [R8-54]
):
    """Returns a probability-like detection score used downstream."""
    flag_twtd = _USE_TWTD if use_twtd is None else bool(use_twtd)
    g_proc = _G_PROC if G_proc is None else float(G_proc)

    # SNR factor in [0,1)
    F_SNR = 1 - math.exp(-kappa_snr * (s_proc / max(s_th, 1e-6)))

    # CFAR amplification: enabled only in TWTD mode (LOS -> neutral gain = 1.0)
    F_CFAR = get_cacfar_detection(d, G_proc=g_proc) if flag_twtd else 1.0

    # Environmental attenuation (e.g., walls)
    F_env = math.exp(-beta * max(m, 0.0))

    # Final score (scaled as in your original)
    P_detection = F_SNR * F_env * F_CFAR / 10.0

    # Adversarial attenuation (higher level => lower detection)  [R6-4, R8-36]
    if _ADV_LVL > 0.0:
        P_detection *= (1.0 - 0.5*_ADV_LVL)

    # Optional obstacle attenuation (kept off by default)
    # if obstacle:
    #     P_detection *= obstacle_attenuation

    return P_detection

def detection_probability(distance, point, obstacles_file='experiments/exp-a/env1.json'):
    """Compatibility wrapper for env usage (reads globals)."""
    obstacles = load_obstacles(obstacles_file)
    obstacle = is_point_in_obstacle(point[0], point[1], obstacles)
    return calculate_detection_probability(distance, obstacle=obstacle)

#
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
#
## Set up the grid for the heat map
## Centers are the location of the target
## Define custom colormap ranging from white to the color '1BA1E2'
#cmap = LinearSegmentedColormap.from_list(
#    'custom_blue', ['white', '#1BA1E2'], N=256
#)
#
#center_x, center_y = 27, 10
#grid_size = 10
#x = np.linspace(center_x - 5, center_x + 5, grid_size)
#y = np.linspace(center_y - 5, center_y + 5, grid_size)
#X, Y = np.meshgrid(x, y)
#
## Calculate the detection probability for each point on the grid
#Z = np.zeros_like(X)
#for i in range(grid_size):
#    for j in range(grid_size):
#        Z[i, j] = detection_probability(X[i, j], Y[i, j])
#
## Plot the heat map
#plt.figure(figsize=(10, 8))
#contour = plt.contourf(X, Y, Z, levels=50, cmap=cmap)
#cbar = plt.colorbar(contour)
#
## Set the label for the colorbar with increased size and bold text
#cbar.set_label(r'$D^{(t)}_i$', fontsize=34, fontweight='bold')
#
## Set the font size and weight for the colorbar tick labels directly
#cbar.ax.tick_params(labelsize=26, width=4)
#for label in cbar.ax.get_yticklabels():
#    label.set_fontsize(26)
#    label.set_fontweight('bold')
#
## Set axis labels with increased size and bold text
#plt.xlabel('X Coordinate (m)', fontsize=28, fontweight='bold')
#plt.ylabel('Y Coordinate (m)', fontsize=28, fontweight='bold')
#
## Set the font size and weight for the tick labels on the x and y axes
#plt.tick_params(axis='both', which='major', labelsize=26, width=4)
#plt.xticks(fontsize=26, fontweight='bold')
#plt.yticks(fontsize=26, fontweight='bold')
#
## Optionally, set the title with increased size and bold text
## plt.title('Radar Detection Probability Heat Map (10x10 grid around center (27, 10))', fontsize=16, fontweight='bold')
#
## Save the figure
#plt.savefig('Figures/radar_TWTD_with_obstacle.png')
#
## Display the plot
#plt.show()
