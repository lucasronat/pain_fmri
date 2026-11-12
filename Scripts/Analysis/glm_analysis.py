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
from nilearn.glm.contrasts import expression_to_contrast_vector

#@click.command()
#@click.option('--contrast')
#@click.option('--contrast_name', type=str)
#@click.option('--alpha', type=float)
#@click.option('--height_control', type=str)
#@click.option('--atlas_name', type=str)

def first_level_analyses(data_dir, task_label, space_label, derivatives_folder, output_dir, contrast_dict=None):
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
    for i, model in enumerate(models[:2]):
        model.fit(
            models_run_imgs[i],
            events=models_events[i],
            confounds=models_confounds[i]
        )
        print(models_run_imgs[i])

        for i, design_matrix in enumerate(model.design_matrices_):

            design_columns = design_matrix.columns.to_list()
            design_columns[0] = "LOW_LOW"
            design_columns[1] = "LOW_MOD"
            design_columns[2] = "MOD_MOD"
            design_columns[3] = "HIGH_MOD"
            design_columns[4] = "HIGH_HIGH"
            design_columns[5] = "LOW_CUE"
            design_columns[6] = "MOD_CUE"
            design_columns[7] = "HIGH_CUE"
            design_matrix.columns = design_columns
            
            plot_design_matrix(design_matrix, 
                               output_file=Path(
                                   output_dir,
                                   f"first_level_design_matrix_{model.subject_label}_run{i}.png")
                              )
        for key, value in contrast_dict.items():
            output_file=model.compute_contrast(value)
            output_file.to_filename(
                Path(output_dir,
                     f"sub-{model.subject_label}",
                     f"sub-{model.subject_label}_task-painexpectancy_space-MNI152NLin2009cAsym_contrast-{key}_stat-z_statmap.nii.gz"
                     )
                
                )
            clean_map, threshold = threshold_stats_img(
                output_file, alpha=0.001, height_control="fdr", two_sided=True
            )
                              
            plot_glass_brain(
                clean_map,
                title=f'Nilearn Z map of {key} (FDR p<0.001)',
                plot_abs=False,
                output_file=Path(output_dir,
                                 f"sub-{model.subject_label}",
                                 f"sub-{model.subject_label}_task-painexpectancy_contrast-{key}_plot.png"),
                display_mode="ortho",
                threshold=threshold,
                figure=plt.figure(figsize=(12, 8)),
            )
        
            html_view = view_img(
                clean_map,
                threshold=threshold,
                dim=0,
            )
            html_view.save_as_html(
                Path(output_dir,
                     f"sub-{model.subject_label}",
                     f"sub-{model.subject_label}_task-painexpectancy_contrast-{key}_plot.html")
            )
            
            table = get_clusters_table(
                clean_map, stat_threshold=threshold, cluster_threshold=10
            )
            table.to_csv(Path(output_dir,
                                 f"sub-{model.subject_label}",
                                 f"sub-{model.subject_label}_task-painexpectancy_contrast-{key}_clusters.csv")
                        )

    return models
        

def second_level_analysis(output_dir, contrast_dict=None):
    """
    Perform second level analysis on first level models.
    """
    for key, value in contrast_dict.items():
        # gather contrasts from first level models
        second_level_input = list(Path(output_dir).rglob(f"*contrast-{key}_stat-z_statmap.nii.gz"))
        print(f"Found {len(second_level_input)} files for contrast {key}")
        design_matrix = pd.DataFrame(
            [1] * len(second_level_input),
            columns=["intercept"],
        )
        
        # instatiate second level model object and fit to first level contrasts
        second_level_model = SecondLevelModel(smoothing_fwhm=9.0, n_jobs=2)
        second_level_model.fit(second_level_input, design_matrix=design_matrix)

        # compute second level contrast and save to file
        z_map = second_level_model.compute_contrast(second_level_contrast="intercept", output_type='z_score')
        z_map.to_filename(
            Path(output_dir, f"second_level_contrast-{key}_stat-z_statmap.nii.gz")
        )
        
        clean_map, threshold = threshold_stats_img(
            z_map, alpha=0.001, height_control="fdr", two_sided=True
        )
        
        plot_glass_brain(
            clean_map,
            threshold=threshold,
            title=f"{key} (FDR p<0.001)",
            plot_abs=False,
            output_file=Path(output_dir,
                                 f"2ndlevelGLM_task-painexpectancy_contrast-{key}_plot_glass.png"),
            display_mode="ortho",
            figure=plt.figure(figsize=(12, 8)),
        )
        
        plot_stat_map(
            clean_map,
            display_mode="ortho",
            threshold=threshold,
            output_file=Path(output_dir,
                                 f"2ndlevelGLM_task-painexpectancy_contrast-{key}_plot_stat_map.png"),
            title=f"{key} (FDR p<0.001)",
        )
        
        html_view = view_img(
            clean_map,
            threshold=threshold,
            dim=0,
        )
        html_view.save_as_html(Path(output_dir,
                                    f"2ndlevelGLM_task-painexpectancy_contrast-{key}_plot.html")
                              )
            
        table = get_clusters_table(
            clean_map, stat_threshold=threshold, cluster_threshold=10
        )
        table.to_csv(Path(output_dir,
                     f"2ndlevelGLM_task-painexpectancy_contrast-{key}_clusters.csv")
                    )
        # save report of second level analysis
        report = second_level_model.generate_report()
        report.save_as_html(
            Path(output_dir, f"second_level_contrast-{key}_report.html")
        )
    return

def main():
    data_dir = Path('/scratch/lronat/pdpain')
    task_label = "painexpectancy"
    space_label = "MNI152NLin2009cAsym"
    derivatives_folder = "derivatives/tedana"
    output_dir = Path("/home/lronat/pdpain/results/plot_bids_features/derivatives/nilearn_glm")

    # verify that contrast dictionary is valid
    expression_to_contrast_vector(
        "LOW_LOW + LOW_MOD + MOD_MOD + HIGH_MOD + HIGH_HIGH",
        [
            "LOW_LOW",
            "LOW_MOD",
            "MOD_MOD",
            "HIGH_MOD",
            "HIGH_HIGH",
            "LOW_CUE",
            "MOD_CUE",
            "HIGH_CUE",
         ]
    )

    contrast_dict = {
        "pain_high-low": "HIGH_HIGH - LOW_LOW",
        "cue_high-low": "HIGH_CUE - LOW_CUE",
        "hypoalgesia": "LOW_MOD - MOD_MOD",
        "hyperalgesia": "HIGH_MOD - MOD_MOD",
        "expectation": "LOW_MOD + HIGH_MOD",
        "cue": "LOW_MOD + MOD_CUE + HIGH_MOD",
        "pain": "LOW_LOW + LOW_MOD + MOD_MOD + HIGH_MOD + HIGH_HIGH",
    }

    first_level_models = first_level_analyses(
        data_dir, task_label, space_label, derivatives_folder, output_dir, contrast_dict=contrast_dict
    )
    second_level_analysis(output_dir, contrast_dict=contrast_dict)
    
if __name__ == '__main__':
    main()
