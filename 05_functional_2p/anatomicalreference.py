"""
=============================================================================
ANATOMICAL ORIENTATION REFERENCE
=============================================================================

Creates reference figure showing anatomical axes and orientations
for the zebrafish hindbrain imaging data.

Author: Inês Marques
Date: February 2025
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from pathlib import Path

# Output
OUT_DIR = r"C:\Users\OSVALDO\Downloads\2P_PLANE_ANALYSIS"
Path(OUT_DIR, "figures").mkdir(parents=True, exist_ok=True)

# Create figure
fig = plt.figure(figsize=(16, 10))

# =============================================================================
# PANEL A: DORSAL VIEW (XY plane) - Top-down view
# =============================================================================
ax1 = plt.subplot(2, 2, 1)

# Brain outline (simplified hindbrain)
brain_x = [100, 100, 200, 300, 400, 500, 600, 600, 500, 400, 300, 200, 100]
brain_y = [250, 200, 150, 150, 150, 150, 150, 200, 250, 250, 250, 250, 250]

ax1.fill(brain_x, brain_y, color='lightgray', alpha=0.3, edgecolor='black', linewidth=2)

# Midline (HORIZONTAL line at Y = 215.8)
midline_y = 215.8
ax1.axhline(midline_y, color='red', linestyle='--', linewidth=3, label='Midline')

# LEFT hemisphere (ABOVE midline, lower Y values)
ax1.fill([100, 100, 600, 600, 500, 400, 300, 200],
         [150, midline_y, midline_y, 150, 150, 150, 150, 150],
         color='blue', alpha=0.2)
ax1.text(350, 180, 'LEFT', fontsize=20, fontweight='bold', ha='center',
         color='blue')

# RIGHT hemisphere (BELOW midline, higher Y values)
ax1.fill([100, 100, 200, 300, 400, 500, 600, 600],
         [250, midline_y, midline_y, 250, 250, 250, 250, 250],
         color='orange', alpha=0.2)
ax1.text(350, 240, 'RIGHT', fontsize=20, fontweight='bold', ha='center',
         color='orange')

# Cerebellum region
cerebellum = Rectangle((390, 150), 210, 60, 
                       linewidth=3, edgecolor='green', 
                       facecolor='green', alpha=0.1,
                       linestyle='--')
ax1.add_patch(cerebellum)
ax1.text(495, 180, 'Cerebellum\n(approx)', fontsize=12, ha='center',
         va='center', style='italic', color='green', fontweight='bold')

# Rostro-Caudal axis arrow
arrow1 = FancyArrowPatch((100, 120), (600, 120),
                        arrowstyle='<->', mutation_scale=30,
                        linewidth=3, color='purple')
ax1.add_patch(arrow1)
ax1.text(100, 95, 'CAUDAL\n(Posterior)', fontsize=14, ha='center',
         fontweight='bold', color='purple')
ax1.text(600, 95, 'ROSTRAL\n(Anterior)', fontsize=14, ha='center',
         fontweight='bold', color='purple')

# X-axis annotation
ax1.text(100, 130, 'X = 0', fontsize=10, ha='center', color='purple')
ax1.text(600, 130, 'X = 720', fontsize=10, ha='center', color='purple')

# Y-axis (midline) annotation
ax1.text(620, midline_y, f'Y = {midline_y}\n(Midline)', fontsize=10, 
         ha='left', va='center', color='red', fontweight='bold')

# Axes
ax1.set_xlim(50, 650)
ax1.set_ylim(80, 300)
ax1.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y Position (pixels)', fontsize=12, fontweight='bold')
ax1.set_title('A. DORSAL VIEW (Top-Down)\nLooking from above the fish',
             fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(alpha=0.3)
ax1.invert_yaxis()  # Origin at top

# Add coordinate system indicator
ax1.text(0.95, 0.05, 'Dorsal View\n(XY plane, Z varies)',
         transform=ax1.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# =============================================================================
# PANEL B: SAGITTAL VIEW (XZ plane) - Side view
# =============================================================================
ax2 = plt.subplot(2, 2, 2)

# Brain outline (side view)
brain_x2 = [100, 100, 600, 600, 500, 400, 300, 200, 100]
brain_z = [0, 180, 180, 0, 0, 0, 0, 0, 0]

ax2.fill(brain_x2, brain_z, color='lightgray', alpha=0.3, 
         edgecolor='black', linewidth=2)

# Dorsal-Ventral gradient
for z in range(0, 180, 10):
    intensity = 1 - (z / 180)
    ax2.axhspan(z, z+10, color='cyan', alpha=intensity*0.3)

# Cerebellum region
cerebellum2 = Rectangle((390, 60), 210, 20,
                        linewidth=3, edgecolor='green',
                        facecolor='green', alpha=0.15,
                        linestyle='--')
ax2.add_patch(cerebellum2)
ax2.text(495, 70, 'Cb', fontsize=12, ha='center', va='center',
         color='green', fontweight='bold')

# Rostro-Caudal axis
arrow2 = FancyArrowPatch((100, 200), (600, 200),
                        arrowstyle='<->', mutation_scale=30,
                        linewidth=3, color='purple')
ax2.add_patch(arrow2)
ax2.text(100, 220, 'CAUDAL', fontsize=12, ha='center',
         fontweight='bold', color='purple')
ax2.text(600, 220, 'ROSTRAL', fontsize=12, ha='center',
         fontweight='bold', color='purple')

# Dorso-Ventral axis
arrow3 = FancyArrowPatch((650, 0), (650, 180),
                        arrowstyle='<->', mutation_scale=30,
                        linewidth=3, color='teal')
ax2.add_patch(arrow3)
ax2.text(680, 0, 'DORSAL', fontsize=12, ha='left', va='center',
         fontweight='bold', color='teal', rotation=-90)
ax2.text(680, 180, 'VENTRAL', fontsize=12, ha='left', va='center',
         fontweight='bold', color='teal', rotation=-90)

# Z-plane annotations
ax2.text(50, 0, 'Z = 0', fontsize=10, ha='right', color='teal')
ax2.text(50, 90, 'Z = 90', fontsize=10, ha='right', color='teal')
ax2.text(50, 180, 'Z = 180', fontsize=10, ha='right', color='teal')

ax2.set_xlim(50, 720)
ax2.set_ylim(240, -40)
ax2.set_xlabel('X Position (pixels)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Z Plane', fontsize=12, fontweight='bold')
ax2.set_title('B. SAGITTAL VIEW (Side View)\nLooking from the side',
             fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

ax2.text(0.95, 0.95, 'Sagittal View\n(XZ plane)',
         transform=ax2.transAxes, fontsize=10, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# =============================================================================
# PANEL C: CORONAL VIEW (YZ plane) - Front view
# =============================================================================
ax3 = plt.subplot(2, 2, 3)

# Brain outline (front view)
brain_y3 = [150, 150, 250, 250, 150]
brain_z3 = [0, 180, 180, 0, 0]

ax3.fill(brain_y3, brain_z3, color='lightgray', alpha=0.3,
         edgecolor='black', linewidth=2)

# Left hemisphere
ax3.fill([150, 150, 200, 200], [0, 180, 180, 0],
         color='blue', alpha=0.2)
ax3.text(175, 90, 'L', fontsize=20, fontweight='bold', ha='center',
         color='blue')

# Right hemisphere
ax3.fill([200, 200, 250, 250], [0, 180, 180, 0],
         color='orange', alpha=0.2)
ax3.text(225, 90, 'R', fontsize=20, fontweight='bold', ha='center',
         color='orange')

# Midline
ax3.axvline(200, color='red', linestyle='--', linewidth=3, label='Midline')

# Dorso-Ventral axis
arrow4 = FancyArrowPatch((270, 0), (270, 180),
                        arrowstyle='<->', mutation_scale=30,
                        linewidth=3, color='teal')
ax3.add_patch(arrow4)
ax3.text(300, 0, 'DORSAL', fontsize=12, ha='left', va='center',
         fontweight='bold', color='teal', rotation=-90)
ax3.text(300, 180, 'VENTRAL', fontsize=12, ha='left', va='center',
         fontweight='bold', color='teal', rotation=-90)

# Left-Right axis
arrow5 = FancyArrowPatch((150, 220), (250, 220),
                        arrowstyle='<->', mutation_scale=30,
                        linewidth=3, color='brown')
ax3.add_patch(arrow5)
ax3.text(150, 240, 'LEFT', fontsize=12, ha='center',
         fontweight='bold', color='brown')
ax3.text(250, 240, 'RIGHT', fontsize=12, ha='center',
         fontweight='bold', color='brown')

ax3.set_xlim(120, 320)
ax3.set_ylim(260, -40)
ax3.set_xlabel('Y Position (pixels)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Z Plane', fontsize=12, fontweight='bold')
ax3.set_title('C. CORONAL VIEW (Front View)\nLooking at the fish face-on',
             fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=11)
ax3.grid(alpha=0.3)

ax3.text(0.95, 0.95, 'Coronal View\n(YZ plane)',
         transform=ax3.transAxes, fontsize=10, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# =============================================================================
# PANEL D: SUMMARY AND KEY DEFINITIONS
# =============================================================================
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

summary_text = """
ANATOMICAL ORIENTATION SUMMARY

AXES DEFINITIONS:

X-AXIS (0 → 720 pixels):
  • Rostro-Caudal axis
  • 0 = CAUDAL (posterior, tail end)
  • 720 = ROSTRAL (anterior, head end)
  • Cerebellum ≈ X: 390-600

Y-AXIS (0 → 464 pixels):
  • Left-Right (Laterality) axis
  • MIDLINE: Y = 215.8 pixels
  • LEFT: Y < 215.8 (lower Y values)
  • RIGHT: Y > 215.8 (higher Y values)

Z-AXIS (0 → 180 planes):
  • Dorso-Ventral axis
  • 0 = DORSAL (top of brain)
  • 180 = VENTRAL (bottom of brain)
  • 1 μm spacing between planes

LATERALITY (Left/Right):
  • Defined by distance from MIDLINE
  • Midline: Y = 215.8 pixels (horizontal line)
  • LEFT hemisphere: Y < 215.8
  • RIGHT hemisphere: Y > 215.8
  
IMPORTANT REGIONS:
  • Hindbrain: Rhombomeres 2-7
  • Cerebellum: Z ≈ 60-80, X ≈ 390-600
  • Oculomotor nucleus: More rostral
  • Inferior olive: More caudal

IMAGING PARAMETERS:
  • 180 Z-planes (1 μm steps)
  • 250 frames per plane
  • 2 Hz volumetric rate
  • Total: ~7.5 hours recording
"""

ax4.text(0.5, 0.5, summary_text, ha='center', va='center',
         fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
         transform=ax4.transAxes)

ax4.set_title('D. ANATOMICAL DEFINITIONS', fontsize=14, fontweight='bold',
             pad=20)

# =============================================================================
# MAIN TITLE
# =============================================================================
plt.suptitle('Anatomical Orientation Reference: Zebrafish Hindbrain Imaging',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
save_path = Path(OUT_DIR) / "figures" / "anatomical_orientation_reference.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✓ Saved anatomical reference: {save_path}")

# =============================================================================
# CREATE CORRECTION NOTE
# =============================================================================

correction_text = """
=============================================================================
ANATOMICAL ORIENTATION CORRECTION
=============================================================================

ISSUE IDENTIFIED:
The previous laterality analysis incorrectly interpreted the left/right 
orientation based on the X-axis position.

CORRECT ORIENTATION:
- The MIDLINE runs along the X-axis (horizontally in dorsal view)
- The midline Y-coordinate (215.8) divides left from right
- LEFT hemisphere: Y < 215.8 (lower Y values)
- RIGHT hemisphere: Y > 215.8 (higher Y values)

ROSTRO-CAUDAL AXIS:
- X = 0: CAUDAL (posterior)
- X = 720: ROSTRAL (anterior)
- This axis runs horizontally in dorsal view

DORSO-VENTRAL AXIS:
- Z = 0: DORSAL (top)
- Z = 180: VENTRAL (bottom)
- This is the imaging plane direction

The anatomical_orientation_reference.png figure clarifies all axes.

=============================================================================
"""

correction_path = Path(OUT_DIR) / "ORIENTATION_CORRECTION_README.txt"
with open(correction_path, 'w') as f:
    f.write(correction_text)

print(f"✓ Saved correction note: {correction_path}")
print("\n" + "="*70)
print("ANATOMICAL REFERENCE CREATED")
print("="*70)
print("\nPlease review:")
print(f"  1. {save_path}")
print(f"  2. {correction_path}")
print("\nThe plane-by-plane analysis script is ready to run.")
print("="*70)