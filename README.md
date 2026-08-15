# ComfyUI Davcha Sampling Tricks

**Davcha Sampling Tricks** is an advanced custom node extension for ComfyUI. It provides powerful frequency-domain noise manipulation tools (Spectral Blending & EQ), spatial masking capabilities, and a dynamic multi-sampler chaining system. 

Whether you want fine-grained control over the initial noise structure for specific frequencies, target custom noise to specific spatial regions, or swap out sampling algorithms mid-generation, this toolkit gives you the flexibility you need.

## 🌟 Key Features

### 1. Spectral Noise Manipulation (FFT)
Instead of blending noise purely in the spatial domain, these nodes use Fast Fourier Transforms (FFT) to convert noise into the frequency domain. This allows you to apply radial masks and equalizer curves to low, mid, and high frequencies independently.

### 2. Spatial Noise Masking (Energy-Preserving)
Target specific areas of your image with different noise structures using standard masks. Our spatial blending uses an energy-preserving `sqrt()` algorithm to prevent the "variance dip" that usually causes flat, halo-like artifacts along feathered mask edges.

### 3. NestedTensor & Next-Gen Model Support
Designed with modern architectures in mind. The noise generation safely handles `NestedTensor` formats (essential for models like **Minimax H3**) by using non-destructive unbinding and exact memory layout preservation, while remaining fully compatible with standard models like **Flux** and **SDXL**.

---

## 🧩 Included Nodes

### 🎛️ Spectral Noise Blend (`DavchaSpectralNoiseBlend`)
Blends two different noise sources together using a frequency-based radial mask.
* **Inputs:**
  * `noise_a`: First noise source (e.g., Fractal noise).
  * `noise_b`: Second noise source (e.g., Uniform noise).
  * `spectrum`: A 1D tensor controlling the frequency blending curve.
* **How it works:** The `spectrum` tensor maps frequencies from the center (low freq) to the edges (high freq). A value of `0.0` uses pure **Noise B**, while `1.0` uses pure **Noise A**. Smoothstep interpolation ensures seamless transitions between frequency bins.

### 🎚️ Spectral Noise EQ (`DavchaSpectralNoiseEQ`)
Acts as a graphic equalizer for a single noise source. Shape the structural frequencies of your initial noise before sampling begins.
* **Inputs:** 
  * `base_noise`: The starting noise (Standard Uniform Noise is recommended).
  * `spectrum`: A 1D tensor representing your EQ curve.
* **How it works:** 
  * `1.0` = Unity (Unchanged)
  * `0.0` = Mute (Removes the frequency)
  * `> 1.0` = Boost (Amplifies the frequency)
  * *Note: The node automatically applies a global normalization at the end to ensure the overall noise energy remains acceptable for the diffusion scheduler.*

### 🎭 Masked Noise Blend (`DavchaMaskedNoise`)
Spatially blends two noises together using an image mask, perfect for inpainting or localized texture control.
* **Inputs:**
  * `base_noise`: Noise used where the mask is BLACK (`0.0`).
  * `masked_noise`: Noise used where the mask is WHITE (`1.0`).
  * `mask`: Standard ComfyUI mask tensor.
* **How it works:** The node automatically resizes standard pixel masks (e.g., 1024x1024) to match the internal latent grid (e.g., 128x128). It then blends the two noises using energy-preserving math to keep the noise contrast punchy and seamless, even across soft gradient masks.

### ⏳ Scheduled Sampler (`DavchaScheduledSampler`)
Allows you to chain multiple distinct samplers together during a single generation process.
* **Inputs:**
  * `sampler_1`, `sampler_2`, etc.: Autogrowing inputs allowing you to plug in anywhere from 1 to 10 samplers.
  * `splits`: A comma-separated string of integers defining the step transitions.
* **How it works:** If you run 30 total steps and provide 3 samplers, you can set `splits` to `10, 20`. 
  * Steps `0-10`: Uses Sampler 1
  * Steps `11-20`: Uses Sampler 2
  * Steps `21-30`: Uses Sampler 3

---

## 🛠️ Installation

1. Navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository (or place the extension folder here):
   ```bash
   git clone <your-repo-url-here>
   ```
3. Restart ComfyUI. The nodes will be available in the node menu under:
   * `davcha/noise/spectral`
   * `sampling/custom_sampling/samplers`

## 💡 Usage Tips

* **Spectrum Tensors:** The `spectrum` inputs on the noise nodes expect a 1D tensor (a list of float values). You can create these using any standard tensor or float-list generation nodes in ComfyUI.
* **Splits Formatting:** When using the **Scheduled Sampler**, ensure your `splits` string only contains numbers and commas (e.g., `15, 30`). The number of splits should logically align with the number of samplers you've connected minus one.
* **Dtypes:** The spectral nodes perform their FFT math in `float32` for precision but will strictly cast the output back to your latent's original exact dtype (e.g., `float16` or `bfloat16`), preventing upstream dtype crash issues in ComfyUI workflows.