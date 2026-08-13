from comfy_api.latest import io
from comfy.samplers import KSAMPLER
from .spectral_noise import SpectralNoise, SpectralNoiseEQ
from .scheduled_sampler import sampler_function

class DavchaSpectralNoiseBlend(io.ComfyNode):
    """
    Blends a Fractal (A) and Uniform (B) noise together using a frequency-based mask.
    0.0 uses pure Noise B. 1.0 uses pure Noise A.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaSpectralNoiseBlend",
            display_name="Spectral Noise Blend",
            category="davcha/noise/spectral",
            inputs=[
                io.Noise.Input("noise_a"),
                io.Noise.Input("noise_b"),
                io.Sigmas.Input("spectrum", tooltip="1D tensor controlling frequency blending. 0.0 = Noise B, 1.0 = Noise A"),
            ],
            outputs=[
                io.Noise.Output(),
            ]
        )

    @classmethod
    def execute(cls, noise_a, noise_b, spectrum) -> io.NodeOutput:
        blended_noise = SpectralNoise(noise_a, noise_b, spectrum)
        return io.NodeOutput(blended_noise)
    
class DavchaSpectralNoiseEQ(io.ComfyNode):
    """
    Applies a graphic EQ to a single noise source. 
    1.0 is unchanged. 0.0 removes the frequency. >1.0 boosts it.
    """
    
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DavchaSpectralNoiseEQ",
            display_name="Spectral Noise EQ",
            category="davcha/noise/spectral",
            inputs=[
                io.Noise.Input("base_noise", tooltip="Standard Uniform Noise recommended."),
                io.Sigmas.Input("spectrum", tooltip="1D tensor EQ curve. 1.0 = Unity, 0.0 = Mute, 2.0+ = Boost"),
                io.Boolean.Input("normalize", default=True, tooltip="Normalize the output noise to have a standard deviation of 1.0. This is recommended for most use cases."),
            ],
            outputs=[
                io.Noise.Output(),
            ]
        )

    @classmethod
    def execute(cls, base_noise, spectrum, normalize) -> io.NodeOutput:
        eq_noise = SpectralNoiseEQ(base_noise, spectrum, normalize)
        return io.NodeOutput(eq_noise)
        
class DavchaScheduledSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(input=io.Sampler.Input("sampler"), prefix="sampler", min=1, max=10)
        return io.Schema(
            node_id="DavchaScheduledSampler",
            category="sampling/custom_sampling/samplers",
            inputs=[
                io.Autogrow.Input("autogrow", template=template),
                io.String.Input("splits", multiline=False, dynamic_prompts=False),
            ],
            outputs=[
                io.Sampler.Output(),
            ]
        )
   
    @classmethod
    def execute(cls, autogrow, splits) -> io.NodeOutput:
        splits_list = [int(s.strip()) for s in splits.split(',')]
        samplers = [v for k, v in sorted(autogrow.items())]
        
        return io.NodeOutput(KSAMPLER(sampler_function, extra_options={'samplers': samplers, 'splits': splits_list}))