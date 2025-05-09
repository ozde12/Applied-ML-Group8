import pyminiply
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

models_dir = r"C:\Users\ozdep\Documents\aml\Applied-ML-Group8\project_name\models"
output_dir = "data_analysis"
num_models = 15

all_vertices_pos = []
all_normalized_colors = []
all_combined_pca_results = []
all_combined_explained_variance = []
models_loaded_successfully = [False] * num_models
model_has_normals = [False] * num_models
vertex_counts = [0] * num_models
model_labels = [f"Obj_{i+1:02d}" for i in range(num_models)]

aggregated_combined_features = []
aggregated_model_indices = []

try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to '{output_dir}' directory.")
except OSError as e:
    print(f"Error creating directory '{output_dir}': {e}")
    exit(1)

print(f"\nProcessing {num_models} models from '{models_dir}'...\n")

for i in range(num_models):
    model_num = i + 1
    model_label = model_labels[i]
    model_filename = f"obj_{model_num:02d}.ply"
    relative_path_to_ply = os.path.join(models_dir, model_filename)

    print(f"Processing Model {model_num:02d} ({model_filename}): ", end="")

    positions = None
    normals_data = None
    color_data = None
    combined_pca_result = None
    combined_explained_variance = None
    normalized_color = None
    success = False
    has_normals = False

    try:

        vertices_pos, _, normals_data, _, color_data = pyminiply.read(relative_path_to_ply)
        print("Loaded... ", end="")

        current_vertex_count = len(vertices_pos) if vertices_pos is not None else 0
        vertex_counts[i] = current_vertex_count

        if vertices_pos is not None and current_vertex_count > 0:
            positions = vertices_pos
            success = True

            if (normals_data is not None and isinstance(normals_data, np.ndarray) and
                normals_data.shape[0] == current_vertex_count and
                normals_data.shape[1] == 3):
                print("Normals found... ", end="")
                has_normals = True

                norms_magnitude = np.linalg.norm(normals_data, axis=1, keepdims=True)
                zero_norm_mask = norms_magnitude.flatten() == 0
                normals_processed = normals_data.copy()
                if np.any(zero_norm_mask):
                    print(f"Warning: {np.sum(zero_norm_mask)} zero-length normals found. Setting to [0,0,0]. ", end="")
                    normals_processed[zero_norm_mask] = [0, 0, 0]
                    norms_magnitude[zero_norm_mask] = 1.0
                normals_normalized = normals_processed / norms_magnitude

                combined_features = np.hstack((positions, normals_normalized))

                n_features = combined_features.shape[1]
                n_components_pca = min(n_features, current_vertex_count)
                if n_components_pca >= 2:
                    pca = PCA(n_components=n_components_pca)
                    combined_pca_result = pca.fit_transform(combined_features)
                    combined_explained_variance = pca.explained_variance_ratio_
                    print("Combined PCA Done.")

                    aggregated_combined_features.append(combined_features)
                    aggregated_model_indices.append(np.full(current_vertex_count, i, dtype=int))

                else:
                     print("Skipped Combined PCA (data dim < 2).")
                     has_normals = False

            else:
                print("Normals not found or invalid... Skipped Combined PCA.")
                has_normals = False

            if (color_data is not None and isinstance(color_data, np.ndarray) and
                color_data.shape[0] == current_vertex_count):
                 normalized_color = color_data.astype(np.float32) / 255.0
            else:
                 normalized_color = None

        else:
            print("Skipped Processing (no vertex data).")
            success = False
            has_normals = False

    except FileNotFoundError:
        print(f"Failed (File Not Found)")
        vertex_counts[i] = 0; success = False; has_normals = False
    except ValueError as e:
        print(f"Failed (Error reading/unpacking PLY data: {e}).")
        vertex_counts[i] = 0; success = False; has_normals = False
    except Exception as e:
        print(f"Failed (Error: {e})")
        vertex_counts[i] = 0; success = False; has_normals = False

    all_vertices_pos.append(positions)
    all_normalized_colors.append(normalized_color)
    all_combined_pca_results.append(combined_pca_result if has_normals and combined_pca_result is not None else None)
    all_combined_explained_variance.append(combined_explained_variance if has_normals and combined_explained_variance is not None else None)
    models_loaded_successfully[i] = success
    model_has_normals[i] = has_normals

print("\nData Loading and Per-Model PCA Complete.")

print("\nPerforming Global PCA on aggregated data...")

if not aggregated_combined_features:
    print("Skipping Global PCA: No models with valid normals were found.")
else:
    try:
        final_aggregated_features = np.vstack(aggregated_combined_features)
        final_aggregated_model_indices = np.concatenate(aggregated_model_indices)
        print(f"Aggregated {final_aggregated_features.shape[0]} vertices from {len(aggregated_combined_features)} models for Global PCA.")

        n_global_components = min(6, final_aggregated_features.shape[0], final_aggregated_features.shape[1])
        if n_global_components >=2:
            global_pca = PCA(n_components=n_global_components)
            print("Fitting Global PCA...")
            global_pca_result = global_pca.fit_transform(final_aggregated_features)
            global_explained_variance = global_pca.explained_variance_ratio_
            print("Global PCA Fit complete.")

            print("Generating Global PCA plot...")
            try:
                fig_global_pca, ax_global_pca = plt.subplots(figsize=(10, 8))

                cmap_name = 'tab20' if num_models <= 20 else 'viridis'
                cmap = plt.get_cmap(cmap_name)

                scatter = ax_global_pca.scatter(
                    global_pca_result[:, 0],
                    global_pca_result[:, 1],
                    c=final_aggregated_model_indices,
                    cmap=cmap,
                    s=1,
                    alpha=0.4
                )

                ax_global_pca.set_xlabel("Global Principal Component 1 (Pos+Norm)")
                ax_global_pca.set_ylabel("Global Principal Component 2 (Pos+Norm)")
                ax_global_pca.set_title("Global 2D PCA of All Models (Vertices & Normals)")
                ax_global_pca.grid(True, linestyle='--', alpha=0.5)

                var_text_global = 'Global Explained Variance (Pos+Norm):\n'
                total_var_global = 0
                num_pcs_to_show_global = min(len(global_explained_variance), 6)
                for pc_idx in range(num_pcs_to_show_global):
                    var_text_global += f'PC{pc_idx+1}: {global_explained_variance[pc_idx]:.3f}\n'
                    total_var_global += global_explained_variance[pc_idx]
                var_text_global += f'Total (Top {num_pcs_to_show_global}): {total_var_global:.3f}'

                ax_global_pca.text(0.02, 0.02, var_text_global, transform=ax_global_pca.transAxes, fontsize=9,
                                     verticalalignment='bottom', horizontalalignment='left',
                                     bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.6))

                unique_indices_in_plot = np.unique(final_aggregated_model_indices)
                legend_elements = []
                norm = mcolors.Normalize(vmin=0, vmax=num_models-1)

                for model_idx in unique_indices_in_plot:
                    label = model_labels[model_idx]
                    color = cmap(norm(model_idx))
                    proxy = plt.Line2D([0], [0], marker='o', color='w', label=label,
                                     markerfacecolor=color, markersize=6, linestyle='None')
                    legend_elements.append(proxy)

                ax_global_pca.legend(handles=legend_elements, title="Models",
                                     bbox_to_anchor=(1.04, 1), loc='upper left')

                plt.tight_layout(rect=[0, 0, 0.85, 1])

                save_path_global_pca = os.path.join(output_dir, "global_combined_pca_2d.png")
                plt.savefig(save_path_global_pca, dpi=150)
                plt.close(fig_global_pca)
                print(f"Saved Global Combined PCA plot to '{save_path_global_pca}'")

            except Exception as e:
                print(f"     An error occurred during Global PCA plotting: {e}")
        else:
            print("Skipping Global PCA plot: Aggregated data resulted in < 2 PCA components.")

    except Exception as e:
        print(f"An error occurred during Global PCA calculation: {e}")

print("\nGenerating and saving Vertex Counts plot...")
try:
    fig_counts, ax_counts = plt.subplots(figsize=(12, 7))
    bars = ax_counts.bar(model_labels, vertex_counts, color='skyblue')
    ax_counts.set_xlabel("Model (Class)")
    ax_counts.set_ylabel("Number of Vertices")
    ax_counts.set_title("Vertex Counts per Model (Class Balance)")
    ax_counts.tick_params(axis='x', rotation=60)
    ax_counts.bar_label(bars, fmt='%d', padding=3)
    plt.tight_layout()
    save_path_counts = os.path.join(output_dir, "vertex_counts_summary.png")
    plt.savefig(save_path_counts, dpi=150)
    plt.close(fig_counts)
    print(f"Saved vertex counts plot to '{save_path_counts}'")
except Exception as e:
    print(f"Could not generate or save vertex counts plot: {e}")


print("\nGenerating and saving individual model plots...")
for i in range(num_models):
    model_num = i + 1
    model_label = model_labels[i]
    model_filename_base = f"obj_{model_num:02d}"

    print(f"-- Processing plots for Model {model_num:02d} ({model_label}) --")

    if not models_loaded_successfully[i]:
        print("   Skipping all plots (initial loading failed).")
        continue

    current_vertices = all_vertices_pos[i]
    if current_vertices is not None and current_vertices.shape[0] > 0 and current_vertices.shape[1] >=3:
         print("   Generating coordinate distribution plot...")
         try:
              fig_hist, axes_hist = plt.subplots(1, 3, figsize=(15, 4.5))
              coord_labels = ['X', 'Y', 'Z']
              colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
              for k in range(3):
                   axes_hist[k].hist(current_vertices[:, k], bins=50, color=colors[k], alpha=0.8)
                   axes_hist[k].set_title(f'{coord_labels[k]} Coordinate Distribution')
                   axes_hist[k].set_xlabel('Coordinate Value')
                   axes_hist[k].set_ylabel('Frequency')
                   axes_hist[k].grid(axis='y', linestyle='--', alpha=0.6)
              fig_hist.suptitle(f"Coordinate Distributions for {model_label}", fontsize=14)
              plt.tight_layout(rect=[0, 0.03, 1, 0.95])
              save_path_hist = os.path.join(output_dir, f"{model_filename_base}_coord_dist.png")
              plt.savefig(save_path_hist, dpi=120)
              plt.close(fig_hist)
              print(f"     Saved coordinate distribution plot to '{save_path_hist}'")
         except Exception as e:
              print(f"     An error occurred during coordinate histogram plotting: {e}")
    elif current_vertices is None or current_vertices.shape[0] == 0:
         print("   Skipping coordinate distribution plot: No vertex data.")
    else:
         print(f"   Skipping coordinate distribution plot: Vertex data has fewer than 3 dimensions ({current_vertices.shape[1]}).")

    if model_has_normals[i] and all_combined_pca_results[i] is not None:
        current_pca_result = all_combined_pca_results[i]
        if current_pca_result.shape[1] >= 2:
            print("   Generating 2D Per-Model Combined PCA (Pos+Norm) plot...")
            try:
                fig_pca, ax_pca = plt.subplots(figsize=(8, 7))
                pca_result = current_pca_result
                norm_color = all_normalized_colors[i]
                scatter_colors = norm_color if norm_color is not None else pca_result[:, 0]
                cmap_to_use = None if norm_color is not None else 'viridis'
                scatter = ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], c=scatter_colors, s=5, cmap=cmap_to_use, alpha=0.7)
                ax_pca.set_xlabel("Principal Component 1 (Pos+Norm)")
                ax_pca.set_ylabel("Principal Component 2 (Pos+Norm)")
                ax_pca.set_title(f"2D PCA of Vertices & Normals ({model_label})")
                ax_pca.grid(True, linestyle='--', alpha=0.5)
                ax_pca.set_aspect('equal', adjustable='box')
                if norm_color is None and cmap_to_use is not None:
                     cbar = fig_pca.colorbar(scatter, ax=ax_pca)
                     cbar.set_label('PC1 Value (Color Scale)')
                if all_combined_explained_variance[i] is not None:
                     var_ratio = all_combined_explained_variance[i]
                     var_text = 'Explained Variance (Pos+Norm):\n'; total_var = 0
                     num_pcs_to_show = min(len(var_ratio), 6)
                     for pc_idx in range(num_pcs_to_show):
                         var_text += f'PC{pc_idx+1}: {var_ratio[pc_idx]:.3f}\n'; total_var += var_ratio[pc_idx]
                     var_text += f'Total (Top {num_pcs_to_show}): {total_var:.3f}'
                     ax_pca.text(0.02, 0.02, var_text, transform=ax_pca.transAxes, fontsize=9, va='bottom', ha='left', bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.6))
                plt.tight_layout()
                save_path_pca = os.path.join(output_dir, f"{model_filename_base}_pca_2d.png")
                plt.savefig(save_path_pca, dpi=150)
                plt.close(fig_pca)
                print(f"     Saved 2D Per-Model Combined PCA plot to '{save_path_pca}'")
            except Exception as e:
                print(f"     An error occurred during 2D Per-Model Combined PCA plotting: {e}")
        else:
            print(f"   Skipping Per-Model Combined PCA plot (PCA resulted in < 2 components: {current_pca_result.shape[1]}).")
    else:
         if not model_has_normals[i] and models_loaded_successfully[i]:
              print("   Skipping Per-Model Combined PCA plot (Normals were not found or invalid).")
         elif all_combined_pca_results[i] is None and model_has_normals[i]:
              print("   Skipping Per-Model Combined PCA plot (PCA calculation failed or was skipped earlier).")

print("\nScript Finished. Analysis plots saved in 'data_analysis' directory.")



