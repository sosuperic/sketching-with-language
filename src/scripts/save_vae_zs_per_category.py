"""
There is one SketchRNNVAE model trained per category. We want to
save the z's for the dataset (make an inference pass) for each model.
This is so we can train the InstructionToVAEz and VAEzToInstruction models.

NOTES:
- this script assumes all models are in the same subdirectory (runs/sketchrnn/Mar25_2020/vae_onecat/)


Usage:
PYTHONPATH=. python src/scripts/save_vae_zs_per_category.py
"""
import os
import subprocess

from src.data_manager.quickdraw import final_categories

CMD = """
CUDA_VISIBLE_DEVICES={} PYTHONPATH=. python src/models/sketch_rnn.py --inference_vaez \
--batch_size=8 --max_per_category=70000 --categories={} \
--load_model_path=runs/sketchrnn/Mar25_2020/vae_onecat/encln_KLfix-batch_size_64-categories_{}-dataset_ndjson-dec_dim_2048-enc_dim_512-enc_num_layers_1-lr_0.0001-max_per_category_70000-model_type_vae-use_categories_dec_False-use_layer_norm_True/
"""

def save_zs(gpu=6):

    cats = final_categories()
    processes = []
    for cat in cats:
        cmd = CMD.format(gpu, cat, cat)
        print(cmd)

        # Currently, skip if directly to hold outputs already exists
        out_dir = os.path.join(
            'runs/sketchrnn/Mar25_2020/vae_onecat/encln_KLfix-batch_size_64-categories_{}-dataset_ndjson-dec_dim_2048-enc_dim_512-enc_num_layers_1-lr_0.0001-max_per_category_70000-model_type_vae-use_categories_dec_False-use_layer_norm_True/',
            'inference_vaez/{}'.format(cat))
        if os.path.exists(out_dir):
            print('Files already exist at: ', dir)
            continue

        # Call command
        proc = subprocess.call(cmd, shell=True)  # call is blocking
        print('Done with: ', cat)
        processes.append(proc)

    exit_codes = [p.wait() for p in processes]
    print('Done!')

if __name__ == "__main__":
    save_zs()