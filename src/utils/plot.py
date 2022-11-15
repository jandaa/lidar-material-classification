from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from utils.utils import load_waveforms

plt.style.use("seaborn")
np.random.seed(10)

SHOW_FIG = False
output_dir = Path("plots/multi-pixel")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_materials(waveforms, materials, wave, filename="materials"):

    # Is materials a single type?
    is_single_material = False
    if len(materials) == 1:
        is_single_material = True

    # Plot
    colors = cm.rainbow(np.linspace(0, 1, len(materials)))
    for color_ind, material in enumerate(materials):

        if is_single_material:
            color = None
        else:
            color = colors[color_ind]

        if is_single_material:
            num_samples = 15
        else:
            num_samples = 5

        for i in range(num_samples):

            # Pick a random sample
            waveform = waveforms[material][wave]
            ind = np.random.randint(0, waveform.shape[0])

            # Plot
            label = None
            if i == 0 and not is_single_material:
                label = material

            plt.plot(waveform[ind], label=label, color=color)

    title = f"{wave.capitalize()} Waveform"
    if is_single_material:
        title += " - " + material.capitalize()

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(title)

    if SHOW_FIG:
        plt.show()
    else:
        if is_single_material:
            filename = f"{material}_{wave}.pdf"
        else:
            filename += f"_{wave}.pdf"
        plt.savefig(output_dir / filename, dpi=300)

    plt.clf()


if __name__ == "__main__":

    materials = ["aluminum", "black_cloth", "wood", "black"]
    cardboards = ["black", "blue", "orange", "white", "yellow"]

    preprocessed_path = Path("preprocessed")
    waveforms = load_waveforms(preprocessed_path, materials + cardboards)

    plt.figure(figsize=(8, 6))

    for material in materials:
        plot_materials(waveforms, [material], "low")
        plot_materials(waveforms, [material], "high")

    plot_materials(waveforms, materials, "low", filename="materials")
    plot_materials(waveforms, materials, "high", filename="materials")
    plot_materials(waveforms, cardboards, "low", filename="cardboards")
    plot_materials(waveforms, cardboards, "high", filename="cardboards")
