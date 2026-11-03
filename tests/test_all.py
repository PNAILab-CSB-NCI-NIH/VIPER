import os
import sys
import torch
import unittest
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
from viper.dataset import Dataset
from viper.denoiser import Denoiser
from viper.predictor import Predictor
from viper.analysis import AFMComparison
from viper.crystal import AFMLatticeAnalysis

THIS_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[1]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data_path = str(REPO_ROOT) + "/data/raw"
        self.out_path = "output"

        self.assertTrue(
            os.path.exists(self.data_path),
            f"Missing tutorial data: {self.data_path}",
        )
        os.makedirs(self.out_path, exist_ok=True)
        
    def tearDown(self):
        plt.close("all")

    def test_0_train_pipe(self):
        data = Dataset(self.data_path)
        data.view_image(idx=0)
        data.make_stripe_masks(
            idx=0, k_pos=3.5, k_neg=3.5, row_window=4,
            min_length=10, min_thickness=1, min_area=10,
            output=self.out_path
        )
        data.random_crops(300, 32, 32, drift_mode="per_image")
        data.quality_cut(x_min=-4, x_max=0, y_min=-2, y_max=2)
        data.normalize()

        # Model training
        d = Denoiser(name="N2N_test" , device=DEVICE)
        d.set_dataset(data)
        d.split(fractions=[0.95, 0.05, 0.0], norm=True)
        d.build_model(arch="n2n", lr=0.001 )
        d.train(
            epochs=10, batch_size=32, patience=3, v=2,
            monitor="val_loss", augment=False, output=self.out_path
        )

    def test_1_denoise_pipe(self):
        data = Dataset(self.data_path)
        data.view_image(idx=0)
        data.make_stripe_masks(
            idx=0, k_pos=3.5, k_neg=3.5, row_window=4,
            min_length=10, min_thickness=1, min_area=10,
            output=self.out_path
        )

        # Model predictor
        model_name = "N2N_test"
        crop_size = 32
        stride = crop_size // 2

        # Initialize model
        pred = Predictor(arch="n2n", device=DEVICE, crop_size=crop_size)
        pred.load(f"{self.out_path}/{model_name}.pt")

        # Choose image index for prediction:
        for idx in [0, 1]:
            X = data.imgs_x[idx]     # Trace image
            y = data.imgs_y[idx]     # Retrace image
            yh = pred.predict_image( # Denoised image
                X, stride=stride, batch_size=32)

            # Create plot and save
            save_name = data.names[idx].replace(".txt", f"_{model_name}")
            _ = pred.compare(X, y, yh, output=save_name + ".png")
            pred.write_image(yh, data.apix[idx], save_name + ".txt")

    def test_2_analysis(self):
        raw_file = f"{self.data_path}/overnight_soaking0081-HeightTrace.txt"
        den_file = f"{self.out_path}/denoised/overnight_soaking0081-HeightTrace_N2N_test.txt"
        out_path = f"{self.out_path}/analysis"

        comp = AFMComparison(raw_file=raw_file, denoised_file=den_file, output_dir=out_path)
        comp.load()
        comp.plot_real_space()
        comp.run_fft()
        comp.plot_fft()
        comp.plot_radial_spectrum()
        met = comp.compute_metrics()
        comp.save_all()

    def test_3_crystal(self):
        img_file = f"{self.out_path}/denoised/CRYS4_APO0000-HeightTrace_N2N_test.txt"
        out_path = f"{self.out_path}/autocorr"
        fov_nm = 100.0
        label = "AFM"

        crystal_phases = {
            "AUC topo 1":   dict(c_A=95.1,  a_A=49.3, beta_deg=91.1),
            "BUC topo 1":   dict(c_A=153.6, a_A=50.2, beta_deg=89.7),
            "AUC rev topo": dict(c_A=93.2,  a_A=49.6, beta_deg=90.1),
        }
        ana = AFMLatticeAnalysis(
            file=img_file, output_dir=out_path, fov_nm=fov_nm, data_label=label
        )

        ana.settings.max_acf_vectors_for_lattice = 30
        ana.settings.max_fft_vectors_for_lattice = 25
        ana.settings.max_acf_peaks = 80
        ana.settings.max_fft_peaks = 40
        ana.settings.acf_threshold_sigma = 2.5
        ana.settings.fft_threshold_sigma = 4.0

        ana.load()                 # Load input
        ana.run_fft()              # Run FFT
        pfft = ana.paired_fft      # FFT periodicities (freq space)
        ana.run_autocorrelation()  # Run autocorrelation (real space)
        ana.find_lattice()         # Compare FFT and ACF and find lattice
        lattice = ana.lattice      # v1, v2, beta, coverage, etc...

        _ = ana.plot_real_space()
        _ = ana.plot_fft()
        _ = ana.plot_radial_spectrum()
        _ = ana.plot_autocorrelation()
        _ = ana.plot_fft_reciprocal_lattice()
        _ = ana.plot_fft_vs_acf(
            title_size=22, label_size=18,
            tick_size=16, legend_size=18
        )

        if crystal_phases:
            comp = ana.compare_to_crystal_phases(crystal_phases)
            _ = ana.plot_direct_vs_crystal()
        ana.save_all()

        os.system(f"rm -r {self.out_path}")

if __name__ == "__main__":
    unittest.main()