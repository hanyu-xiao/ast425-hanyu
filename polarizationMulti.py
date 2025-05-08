#!/usr/bin/env python
"""
polarizationMulti.py - Analyzes polarization and Stokes I FITS files for a given target.

This script processes radio astronomy data to create images of a target in polarization intensity,
polarization fraction, and total intensity. It averages data across multiple observations to reduce noise.

Example usage:
    python polarizationMulti.py --target_pix 1630 1725 --search_radius 10 --output_prefix grb_analysis \
        --sbid_list 44780 44857 44918 45060 45086 45416 46350 46419 46492 46554 48611
        
    python polarizationMulti.py --target_pix 944 2281 --search_radius 10 --output_prefix catalog_source1 \
        --sbid_list 44780 44857 44918 45060 45086 45416 46350 46419 46492 46554 48611
        
    python polarizationMulti.py --target_pix 2611 1705 --search_radius 10 --output_prefix catalog_source2 \
        --sbid_list 44780 44857 44918 45060 45086 45416 46350 46419 46492 46554 48611

The script will save plots for each observation and combined results as PNG files.
"""

import argparse
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze polarization data for a target.')
    parser.add_argument('--target_pix', type=int, nargs=2, required=True,
                       help='Target pixel coordinates (y x)')
    parser.add_argument('--search_radius', type=int, default=10,
                       help='Half the side length of the square search region')
    parser.add_argument('--output_prefix', type=str, default='target',
                       help='Prefix for output PNG files')
    parser.add_argument('--sbid_list', type=str, nargs='+', 
                        default=['44780', '44857', '44918', '45060', '45086', 
                                '45416', '46350', '46419', '46492', '46554', '48611'],
                       help='List of scheduling block IDs to process')
    
    args = parser.parse_args()

    # Assign variables from arguments
    targetPix = tuple(args.target_pix)
    searchRadius = args.search_radius
    output_prefix = args.output_prefix
    SBIDlist = args.sbid_list

    # Define target region
    targetRegion = (slice(targetPix[0]-searchRadius, targetPix[0]+searchRadius), 
                   slice(targetPix[1]-searchRadius, targetPix[1]+searchRadius))
    
    # Initialize arrays
    numObs = len(SBIDlist)
    dataPavg = np.zeros((3240, 3240))  # Should match FITS file dimensions
    dataIavg = np.zeros_like(dataPavg)
    targetPlist = np.zeros(numObs)
    targetPnoiselist = np.zeros(numObs)
    targetPfraclist = np.zeros(numObs)
    targetIlist = np.zeros(numObs)

    # Create and save individual observation plots
    fig, axes = plt.subplots(numObs, 3, figsize=(15, 5*numObs))
    if numObs == 1:
        axes = axes.reshape(1, -1)  # Ensure axes is 2D for single observation

    for k, sbid in enumerate(SBIDlist):
        try:
            # Process polarization data
            with fits.open(f'sb{sbid}/results/SB{sbid}-MFS-P-image.fits') as hdulObsP:
                dataObsP = hdulObsP[0].data
            sliceP = dataObsP[targetRegion]
            targetPlist[k] = np.nanmax(sliceP)
            targetPnoiselist[k] = np.nanstd(sliceP)
            dataPavg += dataObsP / numObs

            # Process polarization fraction data
            with fits.open(f'sb{sbid}/results/SB{sbid}-MFS-Pfrac-image.fits') as hdulObsPf:
                dataObsPf = hdulObsPf[0].data
            slicePf = dataObsPf[targetRegion]
            targetPfraclist[k] = np.nanmax(slicePf)

            # Process Stokes I data
            with fits.open(f'sb{sbid}/image/SB{sbid}-MFS-image.fits') as hdulObsI:
                dataObsI = hdulObsI[0].data[0, 0]
            sliceI = dataObsI[targetRegion]
            targetIlist[k] = np.nanmax(sliceI)
            dataIavg += dataObsI / numObs

            # Plot P
            im0 = axes[k, 0].imshow(sliceP, origin='lower', cmap='viridis')
            if np.isnan(targetPfraclist[k]):
                axes[k, 0].text(0.95, 0.95, f'Noise: {targetPnoiselist[k]:.2e}', 
                               transform=axes[k, 0].transAxes, ha='right', va='top', 
                               color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.9))
            axes[k, 0].set_title(f'SB{sbid}: Polarization')
            axes[k, 0].set_xticks([])
            axes[k, 0].set_yticks([])
            plt.colorbar(im0, ax=axes[k, 0])
            
            # Plot Pfrac
            im1 = axes[k, 1].imshow(slicePf, origin='lower', cmap='viridis')
            axes[k, 1].set_title(f'SB{sbid}: Polarization fraction')
            axes[k, 1].set_xticks([])
            axes[k, 1].set_yticks([])
            plt.colorbar(im1, ax=axes[k, 1])

            # Plot Stokes I
            im2 = axes[k, 2].imshow(sliceI, origin='lower', cmap='viridis')
            axes[k, 2].set_title(f'SB{sbid}: Stokes I')
            axes[k, 2].set_xticks([])
            axes[k, 2].set_yticks([])
            plt.colorbar(im2, ax=axes[k, 2], label='Jy/beam')

        except FileNotFoundError as e:
            print(f"Warning: Could not process SB{sbid}: {e}")
            continue

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_individual_observations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print noise statistics
    print('Noise', [f' {targetPnoise:.2e}' for targetPnoise in targetPnoiselist], 'Jy/beam')
    print('Polarized fraction limit', [f' {3*targetPnoise/targetI:.2%}' for targetPnoise, targetI in zip(targetPnoiselist, targetIlist)])
    
    targetPnoiseAvg = np.nanstd(dataPavg[targetRegion])
    targetIAvg = np.nanmax(dataIavg[targetRegion])
    print(f'Combined noise of {targetPnoiseAvg:.2e} Jy/beam with polarized fraction limit {3*targetPnoiseAvg/targetIAvg:.2%}')

    # Create and save combined results plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.tight_layout()
    
    # Polarization plot
    PAvgImage = axes[0].imshow(dataPavg[targetRegion]*1e3, origin='lower', cmap='viridis')
    axes[0].set_xticks([]) 
    axes[0].set_yticks([]) 
    cbar0 = plt.colorbar(PAvgImage, ax=axes[0], fraction=.045)
    cbar0.ax.tick_params(labelsize=12)
    
    # Stokes I plot with noise calculation
    IAvgImage = axes[1].imshow(dataIavg[targetRegion]*1e3, origin='lower', cmap='viridis')
    radius = 6  # Adjust based on source size
    y, x = np.ogrid[:searchRadius*2, :searchRadius*2]
    targetI = dataIavg[targetRegion].copy()
    distance_from_center = np.sqrt((x - searchRadius)**2 + (y - searchRadius)**2)
    targetI[distance_from_center < radius] = np.nan

    print(f'Peak total intensity {np.max(dataIavg[targetRegion])*1e3:.2f} mJy/beam')
    print(f'Peak total intensity noise {np.nanstd(targetI)*1e3:.2f} mJy/beam')

    axes[1].set_xticks([]) 
    axes[1].set_yticks([]) 
    cbar1 = plt.colorbar(IAvgImage, ax=axes[1], fraction=.045, label='mJy/beam')
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label('mJy/beam', size=13)
    
    plt.savefig(f'{output_prefix}_combined_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()