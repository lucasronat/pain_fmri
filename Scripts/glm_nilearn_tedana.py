import warnings
warnings.filterwarnings('ignore')
import bids
import click
import glob
from pathlib import Path
import nilearn
from nilearn.plotting import view_img, plot_glass_brain, show, plot_design_matrix, plot_stat_map
from nilearn.image import load_img, index_img
from nilearn.glm.first_level import first_level_from_bids, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn.glm import save_glm_to_bids, threshold_stats_img
from nilearn.reporting import get_clusters_table
import nibabel as nib
from scipy.stats import norm
from math import ceil
from mni_to_atlas import AtlasBrowser

@click.command()
@click.option('--contrast')
@click.option('--contrast_name', type=str)
@click.option('--alpha', type=float)
@click.option('--height_control', type=str)
@click.option('--atlas_name', type=str)

def main(contrast, contrast_name, alpha, height_control, atlas_name):        
    data_dir = '/scratch/lronat/pdpain'
    task_label = "painexpectancy"
    space_label = "MNI152NLin2009cAsym"
    derivatives_folder = "derivatives/tedana"
    (
        models,
        models_run_imgs,
        models_events,
        models_confounds,
    ) = first_level_from_bids(
        data_dir,
        task_label,
        space_label,
        smoothing_fwhm=9.0,
        derivatives_folder=derivatives_folder,
        n_jobs=2,
        verbose=0,
    )
    
    t_r = 1.48
    n_scans = 485
    
    frame_times = (
        np.arange(n_scans) * t_r
    )  # here are the corresponding frame times
    
    # Next, we simulate 6 motion parameters jointly observed with fMRI acquisitions
    rng = np.random.default_rng(42)
    motion = np.cumsum(rng.standard_normal((n_scans, 6)), 0)
    # The 6 parameters correspond to three translations and three
    # rotations describing rigid body motion
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    
    subjects = ["HA01","HA02","HA03","HA05"]
    #,"HA06","HA07","HA08","HA09","HA10","HA11","HA12","HA13","HA14","HA15","HA16","HA17","HA18","HA19","HA20","HA22","HA23","HA24","HA25","HA26","HA27","HA28","HA29","HA30","HA31","HA32"]
    
    for i, subj in enumerate(subjects):
        events_1 = pd.read_csv(f"/scratch/lronat/pdpain/sub-{subj}/func/sub-{subj}_task-painexpectancy_run-01_events.tsv", sep='\t')
        events_2 = pd.read_csv(f"/scratch/lronat/pdpain/sub-{subj}/func/sub-{subj}_task-painexpectancy_run-02_events.tsv", sep='\t')
        events_3 = pd.read_csv(f"/scratch/lronat/pdpain/sub-{subj}/func/sub-{subj}_task-painexpectancy_run-03_events.tsv", sep='\t')
    
        hrf_model = "glover"
        X1_1 = make_first_level_design_matrix(
            frame_times,
            events_1,
            drift_model="polynomial",
            add_regs=motion,
            add_reg_names=add_reg_names,
            hrf_model=hrf_model,
        )
        X1_2 = make_first_level_design_matrix(
            frame_times,
            events_2,
            drift_model="polynomial",
            add_regs=motion,
            add_reg_names=add_reg_names,
            hrf_model=hrf_model,
        )
        X1_3 = make_first_level_design_matrix(
            frame_times,
            events_3,
            drift_model="polynomial",
            add_regs=motion,
            add_reg_names=add_reg_names,
            hrf_model=hrf_model,
        )
    
        model, imgs, events, confounds = (
            models[i],
            models_run_imgs[i],
            models_events[i],
            models_confounds[i],
        )
        subject = f"sub-{model.subject_label}"
        model.minimize_memory = False  # override default
        
        fig, (axs) = plt.subplots(3,
            figsize=(20, 10), constrained_layout=True
        )
        
        plot_design_matrix(X1_1, axes=axs[0])
        axs[0].set_title("Run 01 Event-related design matrix", fontsize=12)
        plot_design_matrix(X1_2, axes=axs[1])
        axs[1].set_title("Run 02 Event-related design matrix", fontsize=12)
        plot_design_matrix(X1_3, axes=axs[2])
        axs[2].set_title("Run 03 Event-related design matrix", fontsize=12)
        #plt.savefig(f"/home/lronat/pdpain/results/plot_bids_features/derivatives/nilearn_glm/sub-{model.subject_label}/sub-{model.subject_label}_design_matrix.png")
        
        design_columns = [
            f"cond_{int(i):02}" for i in range(len(X1_1.columns))
        ]
        design_columns[0] = "LOW_LOW"
        design_columns[1] = "LOW_MOD"
        design_columns[2] = "MOD_MOD"
        design_columns[3] = "HIGH_MOD"
        design_columns[4] = "HIGH_HIGH"
        design_columns[5] = "LOW_CUE"
        design_columns[6] = "MOD_CUE"
        design_columns[7] = "HIGH_CUE"
        X1_1.columns = design_columns
        
        design_columns = [
            f"cond_{int(i):02}" for i in range(len(X1_2.columns))
        ]
        design_columns[0] = "LOW_LOW"
        design_columns[1] = "LOW_MOD"
        design_columns[2] = "MOD_MOD"
        design_columns[3] = "HIGH_MOD"
        design_columns[4] = "HIGH_HIGH"
        design_columns[5] = "LOW_CUE"
        design_columns[6] = "MOD_CUE"
        design_columns[7] = "HIGH_CUE"
        X1_2.columns = design_columns
        
        design_columns = [
            f"cond_{int(i):02}" for i in range(len(X1_3.columns))
        ]
        design_columns[0] = "LOW_LOW"
        design_columns[1] = "LOW_MOD"
        design_columns[2] = "MOD_MOD"
        design_columns[3] = "HIGH_MOD"
        design_columns[4] = "HIGH_HIGH"
        design_columns[5] = "LOW_CUE"
        design_columns[6] = "MOD_CUE"
        design_columns[7] = "HIGH_CUE"
        X1_3.columns = design_columns
        
        model.fit(imgs, design_matrices=[X1_1,X1_2,X1_3])
    
        design_matrix = model.design_matrices_
        
        contrast_matrix = np.eye(design_matrix[0].shape[1])
        contrasts = {
            column: contrast_matrix[i]
            for i, column in enumerate(design_matrix[0].columns)
        }
        
        contrasts = {
            "pain": (
                contrasts["LOW_LOW"]
                + contrasts["LOW_MOD"]
                + contrasts["MOD_MOD"]
                + contrasts["HIGH_MOD"]
                + contrasts["HIGH_HIGH"]
            ),
            "cue": (
                contrasts["LOW_CUE"]
                + contrasts["MOD_CUE"]
                + contrasts["HIGH_CUE"]
            ),
            "painHighMinusLow": (
                contrasts["HIGH_HIGH"]
                - contrasts["LOW_LOW"]
            ),
            "cueHighMinusLow": (
                contrasts["HIGH_CUE"]
                - contrasts["LOW_CUE"]
            ),
            "hypoalgesia": (
                contrasts["LOW_MOD"]
                - contrasts["MOD_MOD"]
            ),
            "hyperalgesia": (
                contrasts["HIGH_MOD"]
                - contrasts["MOD_MOD"]
            ),
            "expectation": (
                contrasts["LOW_MOD"]
                + contrasts["HIGH_MOD"]
            ),
        }
        
        z_map = model.compute_contrast(contrasts[contrast])
        
        clean_map, threshold = threshold_stats_img(
            z_map, alpha=alpha, height_control=height_control, two_sided=True
        )
        
        plot_glass_brain(
            clean_map,
            threshold=8,
            title=f'Nilearn Z map of {contrast_name} ({height_control} p<{alpha})',
            plot_abs=False,
            display_mode="ortho",
        )
        
        output_dir = "/home/lronat/pdpain/results/plot_bids_features/derivatives/nilearn_glm"
        
        z_map = (output_dir+f"/sub-{model.subject_label}/sub-{model.subject_label}_task-painexpectancy_space-MNI152NLin2009cAsym_contrast-{contrast_name}_stat-z_statmap.nii.gz")
        
        plot_glass_brain(
            clean_map,
            title=f'Nilearn Z map of {contrast_name} ({height_control} p<{alpha})',
            plot_abs=False,
            output_file=output_dir+f"/sub-{model.subject_label}/sub-{model.subject_label}_task-painexpectancy_contrast-{contrast_name}_plot.png",
            display_mode="ortho",
            threshold=8,
            figure=plt.figure(figsize=(12, 8)),
        )
    
        html_view = view_img(
            clean_map,
            threshold=8,
            dim=0,
        )
        html_view.save_as_html(output_dir+f"/sub-{model.subject_label}/sub-{model.subject_label}_task-painexpectancy_contrast-{contrast_name}_plot.html")
        
        table = get_clusters_table(
            clean_map, stat_threshold=threshold, cluster_threshold=10
        )
        
        coordinates = table[["X", "Y", "Z"]].values
    
        # Instantiate the AtlasBrowser class and specify the atlas to use
        atlas = AtlasBrowser(atlas_name)
        
        # Find the brain regions at the MNI coordinates (plotting is optional)
        regions = atlas.find_regions(coordinates, plot=True)
        assert regions
        
        # Project 'undefined' coordinates to the nearest defined atlas region
        projected_coords = atlas.project_to_nearest(coordinates)
        projected_regions = atlas.find_regions(projected_coords)
        assert np.any(projected_coords)
        assert projected_regions
        table["Regions"] = projected_regions
        table.to_csv(output_dir+f"/sub-{model.subject_label}/sub-{model.subject_label}_task-painexpectancy_contrast-{contrast_name}_clusters.csv")

    ncols = 3
    nrows = ceil(len(models) / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 45))
    axes = np.atleast_2d(axes)
    model_and_args = zip(
        models, models_run_imgs, models_events, models_confounds, strict=False
    )
    fig.suptitle(f"Subjects Z map of {contrast_name} ({height_control} p<{alpha})")
    
    for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
        # fit the GLM
        model.fit(imgs, events, confounds)
        # compute the contrast of interest
        z_map = model.compute_contrast(contrasts[contrast])
        clean_map, threshold = threshold_stats_img(
            z_map, alpha=alpha, height_control=height_control, two_sided=True
        )
        plot_glass_brain(
            clean_map,
            threshold=8,
            title=f"sub-{model.subject_label}",
            axes=axes[int(midx / ncols), int(midx % ncols)],
            output_file=output_dir+f"/1stlvlglm_task-painexpectancy_contrast-{contrast_name}_plot.png",
            plot_abs=False,
            colorbar=True,
            display_mode="ortho"
        )
    
    second_level_input = models
    print("second level input done!")
    second_level_model = SecondLevelModel(smoothing_fwhm=9.0, n_jobs=2)
    print("second level model done!")
    second_level_model = second_level_model.fit(second_level_input)
    print("second level model fit done!")
    
    z_map = second_level_model.compute_contrast(first_level_contrast=contrasts[contrast])
    print("z_map done!")
    
    clean_map, threshold = threshold_stats_img(
        z_map, alpha=alpha, height_control=height_control, two_sided=True
    )
    
    plot_glass_brain(
        clean_map,
        threshold=8,
        title=f"{contrast_name} ({height_control} p<{alpha})",
        plot_abs=False,
        output_file=output_dir+f"/2ndlevelGLM_task-painexpectancy_contrast-{contrast_name}_plot_glass.png",
        display_mode="ortho",
        figure=plt.figure(figsize=(12, 8)),
    )
    
    plot_stat_map(
        clean_map,
        display_mode="ortho",
        threshold=8,
        output_file=output_dir+f"/2ndlevelGLM_task-painexpectancy_contrast-{contrast_name}_plot_stat_map.png",
        title=f"{contrast_name} ({height_control} p<{alpha})",
    )
    
    html_view = view_img(
        clean_map,
        threshold=8,
        dim=0,
    )
    html_view.save_as_html(output_dir+f"/2ndlevelGLM_task-painexpectancy_contrast-{contrast_name}_plot.html")
        
    table = get_clusters_table(
        clean_map, stat_threshold=threshold, cluster_threshold=10
    )
        
    coordinates = table[["X", "Y", "Z"]].values

    # Instantiate the AtlasBrowser class and specify the atlas to use
    atlas = AtlasBrowser(atlas_name)
    
    # Find the brain regions at the MNI coordinates (plotting is optional)
    regions = atlas.find_regions(coordinates, plot=True)
    assert regions
    
    # Project 'undefined' coordinates to the nearest defined atlas region
    projected_coords = atlas.project_to_nearest(coordinates)
    projected_regions = atlas.find_regions(projected_coords)
    assert np.any(projected_coords)
    assert projected_regions
    table["Regions"] = projected_regions
    table.to_csv(output_dir+f"/2ndlevelGLM_task-painexpectancy_contrast-{contrast_name}_clusters.csv")

if __name__ == '__main__':
    main()
