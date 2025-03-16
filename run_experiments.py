import os, sys, inspect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import torch.backends.cudnn as cudnn
import itertools
import pandas as pd
from tqdm import tqdm
import argparse


from conformal import *
from utils import *

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("GPU not available, using CPU.")



# --- Figure2 functions (Experiment 1) ---
def plot_figure2(df):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    df['desired coverage (1-α)'] = 1 - df['alpha']
    sns.barplot(x='desired coverage (1-α)', y='desired coverage (1-α)', data=df, alpha=0.3, ax=axs[0],
                edgecolor='k', ci=None)

    bplot = sns.barplot(x='desired coverage (1-α)', y='coverage', hue='predictor', data=df, ax=axs[0],
                         alpha=0.5, ci='sd', linewidth=0.01)
    for patch in bplot.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))
    sns.barplot(x='desired coverage (1-α)', y='size', hue='predictor', data=df, ax=axs[1],
                ci='sd', alpha=0.5, linewidth=0.01)
    sns.despine(top=True, right=True)
    axs[0].set_ylim(ymin=0.85, ymax=1.0)
    axs[0].set_yticks([0.85, 0.9, 0.95, 1])
    axs[0].set_ylabel('empirical coverage')
    axs[1].set_ylabel('average size')
    for ax in axs:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
        ax.legend(fontsize=15, title_fontsize=15)
    axs[1].get_legend().remove()
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    os.makedirs('./outputs/', exist_ok=True)
    plt.savefig('./outputs/barplot-figure2.pdf')
    print("Figure2 saved as ./outputs/barplot-figure2.pdf")

def trial(model, logits, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, naive_bool):
    # Split calibration and validation sets from precomputed logits
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size=bsz, shuffle=False, pin_memory=True)
    # Wrap the model with conformal calibration using the precomputed logits version
    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda,
                                            randomized=randomized, allow_zero_sets=True, naive=naive_bool)
    top1_avg, top5_avg, cvg_avg, sz_avg = validate(loader_val, conformal_model, print_bool=False)
    return top1_avg, top5_avg, cvg_avg, sz_avg

def experiment(modelname, datasetname, datasetpath, model, logits, num_trials, alpha, kreg, lamda,
               randomized, n_data_conf, n_data_val, bsz, predictor):
    naive_bool = predictor == 'Naive'
    # For Naive/APS, no regularization is applied.
    if predictor in ['Naive', 'APS']:
        lamda = 0
    df = pd.DataFrame(columns=["model", "predictor", "alpha", "coverage", "size"])
    for i in tqdm(range(num_trials), desc="Trials"):
        top1_avg, top5_avg, cvg_avg, sz_avg = trial(model, logits, alpha, kreg, lamda,
                                                    randomized, n_data_conf, n_data_val, bsz, naive_bool)
        new_row = pd.DataFrame([{
            "model": modelname,
            "predictor": predictor,
            "alpha": alpha,
            "coverage": cvg_avg,
            "size": sz_avg
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    return df

# --- make_table function (from Experiment 2 / Table2) ---
def make_table(df, alpha):
    round_to_n = lambda x, n: np.round(x, -int(np.floor(np.log10(x))) + (n - 1))
    df = df[df.alpha == alpha]
    table = ""
    table += "\\begin{table}[t]\n"
    table += "\\centering\n"
    table += "\\small\n"
    table += "\\begin{tabular}{lcccccccccc}\n"
    table += "\\toprule\n"
    table += " & \\multicolumn{2}{c}{Accuracy}  & \\multicolumn{4}{c}{Coverage} & \\multicolumn{4}{c}{Size} \\\\ \n"
    table += "\\cmidrule(r){2-3}  \\cmidrule(r){4-7}  \\cmidrule(r){8-11}\n"
    table += "Model & Top-1 & Top-5 & Top K & Naive & APS & RAPS & Top K & Naive & APS & RAPS \\\\ \n"
    table += "\\midrule\n"
    for model in df.Model.unique():
        df_model = df[df.Model == model]
        table += f" {model} & "
        table += f" {np.round(df_model.Top1.mean(), 3)} & "
        table += f" {np.round(df_model.Top5.mean(), 3)} & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "Fixed"].item(), 3)) + " & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "APS"].item(), 3)) + " & "
        coverage_values = df_model.Coverage[df_model.Predictor == "RAPS"].values
        if len(coverage_values) == 1:
            table += str(round_to_n(coverage_values[0], 3)) + " & "
        else:
            table += "N/A & "  # or handle the error in another way

        table += str(round_to_n(df_model["Size"][df_model.Predictor == "Fixed"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "APS"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "RAPS"].item(), 3)) + " \\\\ \n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Results on Imagenet-V2.} \n"
    table += "\\label{table:imagenet-v2}\n"
    table += "\\end{table}\n"
    return table

# --- Figure4 functions (Experiment 3: set-size histograms) ---
def plot_figure4(df_big):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 1.8))
    lamdas_unique = df_big.lamda.unique()
    lamdas_unique.sort()
    for i in range(len(lamdas_unique)):
        lamda = lamdas_unique[i]
        df = df_big[df_big.lamda == lamda]
        d = 1 
        left_of_first_bin = - float(d)/2
        right_of_last_bin = 100 + float(d)/2
        histbins = np.arange(left_of_first_bin, right_of_last_bin + d, d)
        for predictor in ['Naive', 'APS', 'RAPS']:
            to_plot = df['size'][df.predictor == predictor]
            sns.distplot(list(to_plot), bins=histbins, hist=True, kde=False, rug=False,
                         norm_hist=True, label=predictor,
                         hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.5}, ax=axs[i])
        sns.despine(top=True, right=True, ax=axs[i])
        axs[i].set_xlabel('size', fontsize=12)
        axs[i].legend(title='method', framealpha=0.95)
        axs[i].set_yscale('log')
        axs[i].set_yticks([0.1, 0.01, 0.001])
        axs[i].set_ylabel('', fontsize=12)
        axs[i].set_ylim(top=0.5)
        if lamda != lamdas_unique.max():
            axs[i].get_legend().remove()
        axs[i].text(40, 0.07, f'λ={lamda}')
    axs[0].set_ylabel('frequency', fontsize=12)
    plt.tight_layout(rect=[0.03, 0.05, 0.95, 0.93])
    os.makedirs('./outputs/', exist_ok=True)
    plt.savefig('./outputs/noviolin_histograms_figure4.pdf')
    print("Figure4 saved as ./outputs/noviolin_histograms_figure4.pdf")

def sizes_topk(modelname, datasetname, datasetpath, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor):
    naive_bool = predictor == 'Naive'
    lamda_predictor = lamda
    if predictor in ['Naive', 'APS']:
        lamda_predictor = 0
    logits = get_logits_dataset(modelname, datasetname, datasetpath)
    logits_cal, logits_val = split2(logits, n_data_conf, n_data_val)
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(logits_val, batch_size=bsz, shuffle=False, pin_memory=True)
    model = get_model(modelname)
    conformal_model = ConformalModelLogits(model, loader_cal, alpha=alpha, kreg=kreg, lamda=lamda_predictor,
                                            randomized=randomized, allow_zero_sets=True, naive=naive_bool)
    df = pd.DataFrame(columns=['model', 'predictor', 'size', 'topk', 'lamda'])
    for i, (logit, target) in tqdm(enumerate(loader_val), desc="Computing sizes"):
        output, S = conformal_model(logit)
        size = np.array([x.size for x in S])
        I, _, _ = sort_sum(logit.numpy())
        topk = np.where((I - target.view(-1, 1).numpy()) == 0)[1] + 1
        batch_df = pd.DataFrame({'model': modelname, 'predictor': predictor, 'size': size, 'topk': topk, 'lamda': lamda})
        df = pd.concat([df, batch_df], ignore_index=True)
    return df

# --- make_table9 for Experiment 4 (adaptiveness) ---
def make_table9(df, alpha):
    round_to_n = lambda x, n: np.round(x, -int(np.floor(np.log10(x))) + (n - 1))
    df = df[df.alpha == alpha]
    table = ""
    table += "\\begin{table}[t]\n"
    table += "\\centering\n"
    table += "\\small\n"
    table += "\\begin{tabular}{lcccccccccccc}\n"
    table += "\\toprule\n"
    table += " & \\multicolumn{2}{c}{Accuracy}  & \\multicolumn{5}{c}{Coverage} & \\multicolumn{5}{c}{Size} \\\\ \n"
    table += "\\cmidrule(r){2-3}  \\cmidrule(r){4-8}  \\cmidrule(r){9-13}\n"
    table += "Model & Top-1 & Top-5 & Top K & Naive & APS & RAPS & LAC & Top K & Naive & APS & RAPS & LAC \\\\ \n"
    table += "\\midrule\n"
    for model in df.Model.unique():
        df_model = df[df.Model == model]
        table += f" {model} & "
        table += f" {np.round(df_model.Top1.mean(), 3)} & "
        table += f" {np.round(df_model.Top5.mean(), 3)} & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "Fixed"].item(), 3)) + " & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "APS"].item(), 3)) + " & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "RAPS"].item(), 3)) + " & "
        table += str(round_to_n(df_model.Coverage[df_model.Predictor == "LAC"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "Fixed"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "Naive"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "APS"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "RAPS"].item(), 3)) + " & "
        table += str(round_to_n(df_model["Size"][df_model.Predictor == "LAC"].item(), 3)) + " \\\\ \n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\caption{Results on Imagenet-Val with LAC.}\n"
    table += "\\label{table:imagenet-val-lei-wasserman}\n"
    table += "\\end{table}\n"
    return table

###############################################################################
# Merged Experiments
###############################################################################

# Experiment 1: Coverage vs Set Size on Imagenet (using ResNet152)
def experiment1():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    modelname = 'ResNet152'
    alphas = [0.01, 0.05, 0.10]
    predictors = ['Naive', 'APS', 'RAPS']
    params = list(itertools.product(alphas, predictors))
    datasetname = 'Imagenet'
    datasetpath = 'imagenet'  # <-- change this to your ImageNet val directory
    num_trials = 10
    kreg = 5
    lamda = 0.2
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    cudnn.benchmark = True

    model = get_model(modelname)
    logits = get_logits_dataset(modelname, datasetname, datasetpath)
    df = pd.DataFrame(columns=["model", "predictor", "alpha", "coverage", "size"])
    for (alpha, predictor) in params:
        print(f'Model: {modelname} | Desired coverage: {1 - alpha} | Predictor: {predictor}')
        df_exp = experiment(modelname, datasetname, datasetpath, model, logits,
                            num_trials, alpha, kreg, lamda, randomized,
                            n_data_conf, n_data_val, bsz, predictor)
        df = pd.concat([df, df_exp], ignore_index=True)

    # ✅ Save logits cache for Experiment 4
    os.makedirs('./.cache/', exist_ok=True)
    df.to_csv('./.cache/imagenet_df.csv', index=False)
    print("✅ Logits saved to ./cache/imagenet_df.csv")
    
    plot_figure2(df)
    print("Experiment 1 completed.")


# Experiment 2: Coverage vs Set Size on Imagenet-V2
def experiment2():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    datasetname = 'ImagenetV2'
    datasetpath = 'imagenetV2'  
    modelname = 'ResNet152'
    alphas = [0.05, 0.10]
    predictors = ['Fixed', 'Naive', 'APS', 'RAPS']
    params = list(itertools.product([modelname], alphas, predictors))
    num_trials = 10
    kreg = None
    lamda = None
    randomized = True
    n_data_conf = 5000
    n_data_val = 5000
    bsz = 32
    cudnn.benchmark = True

    df = pd.DataFrame(columns=["Model", "Predictor", "Top1", "Top5", "alpha", "Coverage", "Size"])
    for (modelname, alpha, predictor) in params:
        print(f'Model: {modelname} | Desired coverage: {1 - alpha} | Predictor: {predictor}')
        model = get_model(modelname)
        logits = get_logits_dataset(modelname, datasetname, datasetpath)
        # Removed the extra pct_paramtune argument
        out = experiment(modelname, datasetname, datasetpath, model, logits,
                         num_trials, alpha, kreg, lamda, randomized,
                         n_data_conf, n_data_val, bsz, predictor)
        new_row = pd.DataFrame([{
            "Model": modelname,
            "Predictor": predictor,
            "Top1": np.round(out.iloc[0]['coverage'], 3) if 'coverage' in out.columns else 0,
            "Top5": 0,  # Not computed in this experiment
            "alpha": alpha,
            "Coverage": np.round(out.iloc[0]['coverage'], 3),
            "Size": np.round(out.iloc[0]['size'], 3)
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    alpha_table = 0.10  # choose desired alpha
    table_str = make_table(df, alpha_table)
    os.makedirs('./outputs/', exist_ok=True)
    output_fname = f"outputs/imagenetv2results_{alpha_table}".replace('.', '_') + ".tex"
    with open(output_fname, 'w') as f:
        f.write(table_str)
    print(f"Experiment 2 completed. LaTeX table saved as {output_fname}")

# Experiment 3: Histograms of set sizes (varying λ)
def experiment3():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    modelnames = ['ResNet152']
    alphas = [0.1]
    predictors = ['Naive', 'APS', 'RAPS']
    lamdas = [0.01, 0.1, 1]
    params = list(itertools.product(modelnames, alphas, predictors, lamdas))
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'  
    kreg = 5
    randomized = True
    n_data_conf = 20000
    n_data_val = 20000
    bsz = 64
    cudnn.benchmark = True
    df = pd.DataFrame(columns=["model", "predictor", "size", "topk", "lamda"])
    for (modelname, alpha, predictor, lamda) in params:
        print(f'Model: {modelname} | Desired coverage: {1 - alpha} | Predictor: {predictor} | Lambda = {lamda}')
        df_exp = sizes_topk(modelname, datasetname, datasetpath, alpha, kreg, lamda, randomized, n_data_conf, n_data_val, bsz, predictor)
        df = pd.concat([df, df_exp], ignore_index=True)
    plot_figure4(df)
    print("Experiment 3 completed.")

# Experiment 4: Adaptiveness results (merging LAC with other predictors)
def experiment4():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    modelname = 'ResNet152'
    datasetname = 'Imagenet'
    datasetpath = '/scratch/group/ilsvrc/val/'  
    num_trials = 10
    n_data_conf = 30000
    n_data_val = 20000
    bsz = 32
    randomized = True
    cudnn.benchmark = True
    cache_fname = "./.cache/imagenet_df.csv"
    alpha_table = 0.1
    try:
        df = pd.read_csv(cache_fname)
    except:
        print("Please run Experiment 1 first to generate the logits cache!")
        return
    cache_fname_lw = "./.cache/LAC_imagenet_df.csv"
    try:
        df_lw = pd.read_csv(cache_fname_lw)
    except:
        model = get_model(modelname)
        logits = get_logits_dataset(modelname, datasetname, datasetpath)
        df_lw = pd.DataFrame(columns=["Model", "Predictor", "Top1", "Top5", "alpha", "Coverage", "Size"])
        params_lw = list(itertools.product([modelname], [0.10]))
        for (modelname, alpha) in params_lw:
            print(f'Model: {modelname} | Desired coverage: {1 - alpha} | Predictor: LAC')
            out = experiment(modelname, datasetname, datasetpath, model, logits, num_trials, alpha,
                             None, None, randomized, n_data_conf, n_data_val, bsz, "LAC")
            new_row = pd.DataFrame([{
                "Model": modelname,
                "Predictor": "LAC",
                "Top1": np.round(out.iloc[0]['coverage'], 3) if 'coverage' in out.columns else 0,
                "Top5": 0,
                "alpha": alpha,
                "Coverage": np.round(out.iloc[0]['coverage'], 3),
                "Size": np.round(out.iloc[0]['size'], 3)
            }])
            df_lw = pd.concat([df_lw, new_row], ignore_index=True)
        df_lw.to_csv(cache_fname_lw)
    df_all = pd.concat([df, df_lw], ignore_index=True)
    table_str = make_table9(df_all, alpha_table)
    os.makedirs('./outputs/', exist_ok=True)
    output_fname = f"outputs/table9_{alpha_table}".replace('.', '_') + ".tex"
    with open(output_fname, 'w') as f:
        f.write(table_str)
    print(f"Experiment 4 completed. LaTeX table saved as {output_fname}")




def main():
    parser = argparse.ArgumentParser(description="Merged Experiments for Conformal Prediction on ImageNet")
    parser.add_argument("--exp", type=str, default="all", help="Experiment to run: 1, 2, 3, 4 or all")
    args = parser.parse_args()
    if args.exp in ["1", "all"]:
        print("Running Experiment 1: Coverage vs Set Size on Imagenet")
        experiment1()
    if args.exp in ["2", "all"]:
        print("Running Experiment 2: Coverage vs Set Size on Imagenet-V2")
        experiment2()
    if args.exp in ["3", "all"]:
        print("Running Experiment 3: Set Size Histograms")
        experiment3()
    if args.exp in ["4", "all"]:
        print("Running Experiment 4: Adaptiveness (LAC merged results)")
        experiment4()

if __name__ == "__main__":
    main()
