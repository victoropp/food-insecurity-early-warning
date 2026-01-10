"""
Test script to regenerate only Figures 3 and 4 with improved clarity
"""
import sys
from config import BASE_DIR
sys.path.insert(0, str(BASE_DIR))

from generate_stage2_visualizations import load_all_data, create_figure3_hmm_features, create_figure4_dmd_features
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING IMPROVED FIGURES 3 AND 4")
    print("="*80)

    # Load all data
    print("\nLoading data...")
    data = load_all_data()

    # Create Figure 3: HMM Features (IMPROVED)
    print("\n" + "="*80)
    print("Generating Figure 3 (HMM Features)...")
    print("="*80)
    fig3 = create_figure3_hmm_features(data)
    if fig3 is not None:
        plt.close(fig3)
        print("[OK] Figure 3 complete")

    # Create Figure 4: DMD Features (IMPROVED)
    print("\n" + "="*80)
    print("Generating Figure 4 (DMD Features)...")
    print("="*80)
    fig4 = create_figure4_dmd_features(data)
    if fig4 is not None:
        plt.close(fig4)
        print("[OK] Figure 4 complete")

    print("\n" + "="*80)
    print("TEST COMPLETE - CHECK OUTPUT FILES")
    print("="*80)
