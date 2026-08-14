import torch

def create_radial_filter(spectrum, H, W, device):
    Y = torch.linspace(-1, 1, H, device=device)
    X = torch.linspace(-1, 1, W, device=device)
    y, x = torch.meshgrid(Y, X, indexing='ij')
    
    radius = torch.sqrt(x**2 + y**2)
    radius = radius / radius.max()
    
    num_bins = len(spectrum)
    if num_bins == 1:
        return torch.full((H, W), spectrum[0], device=device)
        
    scaled_radius = radius * (num_bins - 1)
    
    lower_idx = scaled_radius.floor().long().clamp(0, num_bins - 1)
    upper_idx = scaled_radius.ceil().long().clamp(0, num_bins - 1)
    weight = scaled_radius - lower_idx 
    
    # Smoothstep transition
    weight = weight * weight * (3.0 - 2.0 * weight)
    
    spectrum = spectrum.to(device)
    lower_val = spectrum[lower_idx]
    upper_val = spectrum[upper_idx]
    
    return torch.lerp(lower_val, upper_val, weight)

class SpectralNoise:
    def __init__(self, a, b, spectrum_tensor):
        self.a = a
        self.b = b
        self.spectrum = spectrum_tensor
        self.seed = 0


    def _process_single_tensor(self, x, y):
        device = x.device
        dtype = x.dtype  # Remember original dtype (usually float16/bfloat16)
        *_, H, W = x.shape
        
        # 1. Cast only the individual unbinded tensors to float32 for the FFT
        x_f = x.to(torch.float32)
        y_f = y.to(torch.float32)
        
        fft_x = torch.fft.fftshift(torch.fft.fft2(x_f, dim=(-2, -1)), dim=(-2, -1))
        fft_y = torch.fft.fftshift(torch.fft.fft2(y_f, dim=(-2, -1)), dim=(-2, -1))
        
        filter_2d = create_radial_filter(self.spectrum, H, W, device)
        view_shape = [1] * (x.ndim - 2) + [H, W]
        mask_nd = filter_2d.view(*view_shape)
        mask_nd = torch.clamp(mask_nd, 0.0, 1.0)
        
        # Energy-preserving blend
        blended_fft = (torch.sqrt(mask_nd) * fft_x) + (torch.sqrt(1.0 - mask_nd) * fft_y)
        
        # Transform back
        shaped = torch.fft.ifft2(torch.fft.ifftshift(blended_fft, dim=(-2, -1)), dim=(-2, -1)).real
        
        # Normalize
        std = shaped.std()
        if std > 0:
            shaped = (shaped - shaped.mean()) / std
        else:
            shaped = shaped - shaped.mean()
            
        # Return cleanly in the exact original dtype
        return shaped.to(dtype)

    def generate_noise(self, input_latent):
        # DO NOT use global .to(torch.float32) here! It destroys NestedTensor padding.
        x = self.a.generate_noise(input_latent)
        y = self.b.generate_noise(input_latent)
        
        if getattr(x, 'is_nested', False):
            # Unbind gives us views into the exact original padded memory buffer.
            # Using xi.copy_ safely overwrites the data while perfectly preserving the memory layout!
            for xi, yi in zip(x.unbind(), y.unbind()):
                xi.copy_(self._process_single_tensor(xi, yi))
            return x
        else:
            return self._process_single_tensor(x, y)
        
class SpectralNoiseEQ:
    def __init__(self, base_noise, spectrum_tensor, normalize):
        self.base_noise = base_noise
        self.spectrum = spectrum_tensor
        self.normalize = normalize
        self.seed = 0

    def _process_single_tensor(self, x):
        device = x.device
        dtype = x.dtype  
        *_, H, W = x.shape
        
        x_f = x.to(torch.float32)
        
        # 1. To the Frequency Domain
        fft_x = torch.fft.fftshift(torch.fft.fft2(x_f, dim=(-2, -1)), dim=(-2, -1))
        
        # 2. Get our EQ Mask (No longer bounded to 0-1!)
        filter_2d = create_radial_filter(self.spectrum, H, W, device)
        view_shape = [1] * (x.ndim - 2) + [H, W]
        eq_mask = filter_2d.view(*view_shape)
        
        # 3. DIRECT SCALING (The EQ step)
        eq_fft = fft_x * eq_mask
        
        # 4. Back to Spatial Domain
        shaped = torch.fft.ifft2(torch.fft.ifftshift(eq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        
        # 5. Global Normalization
        # (This ensures the overall volume of the noise stays acceptable for the scheduler,
        # but the *distribution* of that volume is dictated by your EQ curve).
        std = shaped.std()
        if std > 0 and self.normalize:
            shaped = (shaped - shaped.mean()) / std
        else:
            shaped = shaped - shaped.mean()
            
        return shaped.to(dtype)

    def generate_noise(self, input_latent):
        x = self.base_noise.generate_noise(input_latent)
        
        if getattr(x, 'is_nested', False):
            # Safe injection for Minimax H3
            for xi in x.unbind():
                xi.copy_(self._process_single_tensor(xi))
            return x
        else:
            # Standard for Flux / SDXL
            return self._process_single_tensor(x)