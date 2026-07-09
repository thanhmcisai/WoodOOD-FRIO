"""FRIO corruption family — khop dung ho da dung o RC-A (brittleness diagnosis).
Suite eval: clean/gauss_light/gauss_heavy/blur/bright_contrast/downscale. Mild (cho C2): blur_light/downscale_light.
Tat dinh theo seed cho phan ngau nhien (gauss)."""
import torch
from torchvision import transforms as TT
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
class _AddNoise:
    def __init__(s,sg,gen=None): s.sg=sg; s.gen=gen
    def __call__(s,x): return x+torch.randn(x.shape,generator=s.gen)*s.sg if s.gen is not None else x+torch.randn_like(x)*s.sg
def _mid(name):
    if name=='blur': return [TT.GaussianBlur(7,sigma=3.0)]
    if name=='blur_light': return [TT.GaussianBlur(5,sigma=1.2)]
    if name=='bright_contrast': return [TT.ColorJitter(brightness=0.7,contrast=0.5)]
    if name=='downscale': return [TT.Resize(80),TT.Resize(224)]
    if name=='downscale_light': return [TT.Resize(150),TT.Resize(224)]
    return []
def corrupt_transform(name, res=224, gen=None):
    base=[TT.Resize(res),TT.CenterCrop(res)]; post=[TT.ToTensor(),TT.Normalize(MEAN,STD)]
    if name=='gauss_light': post=post+[_AddNoise(0.15,gen)]
    elif name=='gauss_heavy': post=post+[_AddNoise(0.30,gen)]
    return TT.Compose(base+_mid(name)+post)
SUITE=['clean','gauss_light','gauss_heavy','blur','bright_contrast','downscale']
def corrupt_tensor_mild(x, name):
    """Ap mild corruption tren batch tensor da chuan hoa (cho C2 view suy giam)."""
    import torch.nn.functional as F
    if name=='blur_light':
        k=torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]],device=x.device)/16.0
        k=k.expand(x.shape[1],1,3,3); return F.conv2d(x,k,padding=1,groups=x.shape[1])
    if name=='downscale_light':
        h=x.shape[-1]; return F.interpolate(F.interpolate(x,size=h//2,mode='bilinear',align_corners=False),size=h,mode='bilinear',align_corners=False)
    if name=='defocus_light':
        r=2; yy=torch.arange(-r,r+1).view(-1,1); xx=torch.arange(-r,r+1).view(1,-1); k=((xx**2+yy**2)<=r*r).float(); k=(k/k.sum()).to(x.device).expand(x.shape[1],1,2*r+1,2*r+1); return F.conv2d(x,k,padding=r,groups=x.shape[1])
    if name=='motion_light':
        Lk=5; k=torch.zeros(Lk,Lk); k[Lk//2,:]=1.0/Lk; k=k.to(x.device).expand(x.shape[1],1,Lk,Lk); return F.conv2d(x,k,padding=Lk//2,groups=x.shape[1])
    if name=='pixelate_light':
        h=x.shape[-1]; return F.interpolate(F.interpolate(x,scale_factor=0.4,mode='nearest',recompute_scale_factor=False),size=h,mode='nearest')
    return x


# ===== TIER 1+2: EXTENDED CORRUPTION SUITE (ImageNet-C-style, grouped, severity 1-3) =====
import numpy as _np, io as _io
from PIL import Image as _Im, ImageEnhance as _IE
try:
    import cv2 as _cv2
except Exception:
    _cv2=None

# nhom corruption -> de bao cao theo family (chung minh dac-hieu tan so)
CORRUPT_FAMILIES = {
    'frequency_heldout': ['jpeg_compression','pixelate','motion_blur','defocus_blur'],   # KHONG train -> test generalization
    'frequency_infamily': ['gaussian_blur','downscale_heavy'],                            # cung ho voi train-view
    'noise': ['gauss_noise','shot_noise','impulse_noise'],
    'photometric': ['contrast','brightness','saturate'],                                  # phi-tan-so -> FRIO khong nen chua het
}
EXT_SUITE = [c for v in CORRUPT_FAMILIES.values() for c in v]
_SEV = {  # 3 muc
 'jpeg_compression':[40,22,11],'pixelate':[0.5,0.35,0.22],'motion_blur':[9,13,17],'defocus_blur':[2,4,6],
 'gaussian_blur':[1.0,2.0,3.5],'downscale_heavy':[112,80,56],
 'gauss_noise':[0.06,0.12,0.20],'shot_noise':[60,25,12],'impulse_noise':[0.03,0.07,0.13],
 'contrast':[0.55,0.4,0.28],'brightness':[0.25,0.4,0.55],'saturate':[0.5,0.3,0.15],
}
def _to_np(im): return _np.array(im.convert('RGB'))
def _corrupt_pil(im, name, sev):
    p=_SEV[name][sev-1]; a=_to_np(im)
    if name=='jpeg_compression':
        b=_io.BytesIO(); im.convert('RGB').save(b,'JPEG',quality=int(p)); b.seek(0); return _Im.open(b).convert('RGB')
    if name=='pixelate':
        w,h=im.size; return im.resize((max(1,int(w*p)),max(1,int(h*p))),_Im.BOX).resize((w,h),_Im.NEAREST)
    if name=='downscale_heavy':
        w,h=im.size; return im.resize((int(p),int(p)),_Im.BILINEAR).resize((w,h),_Im.BILINEAR)
    if name=='motion_blur' and _cv2 is not None:
        k=int(p); ker=_np.zeros((k,k),_np.float32); ker[k//2,:]=1.0/k; return _Im.fromarray(_cv2.filter2D(a,-1,ker))
    if name=='defocus_blur' and _cv2 is not None:
        r=int(p); yy,xx=_np.ogrid[-r:r+1,-r:r+1]; ker=((xx**2+yy**2)<=r*r).astype(_np.float32); ker/=ker.sum()
        return _Im.fromarray(_cv2.filter2D(a,-1,ker))
    if name in('motion_blur','defocus_blur'):  # fallback khong co cv2
        return im.filter(__import__('PIL.ImageFilter',fromlist=['GaussianBlur']).GaussianBlur(int(p)))
    if name=='gaussian_blur':
        from PIL import ImageFilter; return im.filter(ImageFilter.GaussianBlur(radius=float(p)))
    if name=='contrast': return _IE.Contrast(im).enhance(float(p))
    if name=='brightness': return _IE.Brightness(im).enhance(1.0+float(p))
    if name=='saturate': return _IE.Color(im).enhance(float(p))
    return im
class _ImgCorrupt:
    def __init__(s,name,sev): s.name=name; s.sev=sev
    def __call__(s,im): return _corrupt_pil(im,s.name,s.sev)
class _NoiseT:
    def __init__(s,name,sev,gen): s.name=name; s.p=_SEV[name][sev-1]; s.gen=gen
    def __call__(s,x):
        if s.name=='gauss_noise': return x+torch.randn(x.shape,generator=s.gen)*s.p
        if s.name=='shot_noise':
            xc=((x*0.229+0.456).clamp(0,3)); return ((torch.poisson(xc*s.p,generator=s.gen)/s.p)-0.456)/0.229
        if s.name=='impulse_noise':
            m=torch.rand(x.shape,generator=s.gen); x=x.clone(); x[m<s.p/2]=-2.0; x[m>1-s.p/2]=2.0; return x
        return x
def corrupt_transform_ext(name, sev=3, res=224, gen=None):
    base=[TT.Resize(res),TT.CenterCrop(res)]; post=[TT.ToTensor(),TT.Normalize(MEAN,STD)]
    if name in ('gauss_noise','shot_noise','impulse_noise'):
        return TT.Compose(base+post+[_NoiseT(name,sev,gen)])
    return TT.Compose(base+[_ImgCorrupt(name,sev)]+post)
