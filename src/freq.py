"""FRIO frequency tools: hfreq energy ratio (C3 degrade estimator) + high/low split (C1)."""
import torch
def _radial_mask(h, w, cutoff, device):
    fy=torch.fft.fftfreq(h,device=device).view(-1,1); fx=torch.fft.fftfreq(w,device=device).view(1,-1)
    r=torch.sqrt(fy**2+fx**2); return (r<=cutoff).float()  # low-pass mask (1 = giu dai thap)
def hfreq_energy_ratio(x, cutoff=0.2):
    """x: (B,C,H,W) chuan hoa. Tra ti le nang luong dai CAO / tong (mien [0,1]). Blur -> thap."""
    X=torch.fft.fft2(x); P=(X.abs()**2); lp=_radial_mask(x.shape[-2],x.shape[-1],cutoff,x.device)
    tot=P.sum(dim=(-1,-2))+1e-8; high=(P*(1-lp)).sum(dim=(-1,-2))
    return (high/tot).mean(dim=1)  # (B,) trung binh kenh
def split_hl(x, cutoff=0.2):
    X=torch.fft.fft2(x); lp=_radial_mask(x.shape[-2],x.shape[-1],cutoff,x.device)
    low=torch.fft.ifft2(X*lp).real; high=torch.fft.ifft2(X*(1-lp)).real; return low, high
def phase_scramble_high(x, cutoff=0.2, gen=None):
    """Xao pha dai cao, giu bien do (pha cau truc texture)."""
    X=torch.fft.fft2(x); mag=X.abs(); ph=X.angle(); hp=1-_radial_mask(x.shape[-2],x.shape[-1],cutoff,x.device)
    rand=(torch.rand(ph.shape,generator=gen,device=x.device)*2-1)*3.14159
    ph2=ph*(1-hp)+rand*hp; return torch.fft.ifft2(mag*torch.exp(1j*ph2)).real
