import os
import sys
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt


def load_data(data_dir, sample_size):
    print(f"Loading data from {data_dir}")

    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.endswith('.csv')]

    if not data_files:
        print("No files found in data directory!")
        sys.exit(1)

    if sample_size == -1:
        sample_files = data_files
    if sample_size < 1.0 and sample_size > 0.0:
        sample_files = data_files[:int(len(data_files) * sample_size)]
    else:
        sample_files = data_files[:sample_size]
    data_list = []
    feature_names = []

    for file_path in sample_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            data_start = -1
            for i, line in enumerate(lines):
                if 'omega, real, im' in line.lower():
                    data_start = i + 1
                    break

            if data_start == -1:
                print(f"  No data section found in {file_path}")
                continue

            reals = []
            imags = []
            omegas = []
            data_lines = 0
            for line in lines[data_start:]:
                line = line.strip()
                if not line or line.startswith('#') or 'omega' in line.lower():
                    continue

                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        omega = float(parts[0])
                        real = float(parts[1])
                        imag = float(parts[2])
                        reals.append(real)
                        imags.append(imag)
                        omegas.append(omega)
                        data_lines += 1
                    except ValueError as ve:
                        print(f"  ValueError parsing line: {line}")
                        continue

            if reals and imags:
                real_names = [f"{format(omegas[i], '.2G')} Hz re" for i in range(len(reals))]
                imag_names = [f"{format(omegas[i], '.2G')} Hz im" for i in range(len(imags))]
                feature_names = real_names + imag_names

                combined = reals + imags
                data_list.append(combined)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not data_list:
        print("No valid data found!")
        sys.exit(1)

    X = np.array(data_list)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

    return X, feature_names


def run_shap_analysis(model, X_sample):
    try:
        background_cutoff = min(100, int(len(X_sample) / 2))
        background_data = X_sample[:background_cutoff]

        def model_wrapper(X):
            Xtorch = torch.tensor(X, dtype=torch.float16, device='cuda:0')
            with torch.no_grad():
                prediction = model(Xtorch).cpu().numpy()
                return prediction

        explainer = shap.KernelExplainer(model_wrapper, background_data)
        test_data = X_sample[background_cutoff + 1:]
        shap_values = explainer.shap_values(test_data, nsamples=2048)

        return shap_values, test_data, explainer
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def plot_shap_results(shap_values, test_data, explainer, feature_names):
    print("Plotting results...")

    try:
        shap.summary_plot(shap_values, test_data, max_display=15, feature_names=feature_names, sort=True, show=False)
        fig = plt.gcf()
        fig.savefig('feature_importance.png')
        plt.close(fig)

        print("plots saved")
    except Exception as e:
        print(f"Plotting Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    model_path = "finished_network/module.pt"
    data_dir = "dataset"

    model = torch.jit.load(model_path)
    model.eval()
    model.to(dtype=torch.float16, device='cuda:0')

    X, feature_names = load_data(data_dir, sample_size=0.5)

    shap_values, test_data, explainer = run_shap_analysis(model, X)
    shap_values = np.squeeze(shap_values, axis=2)
    shap_absmean = np.mean(np.abs(shap_values), axis=0)
    np.savetxt("sharp_absmean.csv", shap_absmean, delimiter=',')

    plot_shap_results(shap_values, test_data, explainer, feature_names)
