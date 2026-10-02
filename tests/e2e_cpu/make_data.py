"""Synthetic fastMRI-style data for the CPU end-to-end test (see README.md). Usage: make_data.py <workdir>"""
import h5py, numpy as np, json, sys
root = sys.argv[1]
rng = np.random.default_rng(0)
H, W, C = 320, 336, 4
yy, xx = np.mgrid[:H, :W]
for i in range(14):
    img = np.zeros((H, W), complex)
    knee = ((yy - H/2) / (120 + 5*i)) ** 2 + ((xx - W/2) / 110) ** 2 < 1
    img[knee] = 1 + 0.3 * np.sin(xx[knee] / (5 + i % 3)) + 0.2 * np.cos(yy[knee] / 9)
    img *= np.exp(1j * 0.002 * (xx - W/2))
    maps = [np.exp(-(((yy - cy) / 200) ** 2 + ((xx - cx) / 200) ** 2)) for cy, cx in [(0, 0), (0, W), (H, 0), (H, W)]]
    coils = np.stack([img * m for m in maps]) + 0.01 * (rng.standard_normal((C, H, W)) + 1j * rng.standard_normal((C, H, W)))
    ksp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coils, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)) * 1e-4
    with h5py.File(f"{root}/data/file{1000+i}.h5", "w") as f:
        f.create_dataset("kspace", data=ksp[None].astype(np.complex64))   # 1 slice
        f.attrs["acquisition"] = "CORPD_FBK"
names = [f"file{1000+i}.h5" for i in range(14)]
sel = f"{root}/results/selections"
json.dump([names[12], names[13], names[8], names[9]], open(f"{sel}/kavg_ref_selection_kavg4.json", "w"))
json.dump([names[12]], open(f"{sel}/kavg_ref_selection_kavg1.json", "w"))
json.dump({"images": [names[10], names[11]], "config": {"old": True}}, open(f"{root}/results/grid_search/grid_v2/manifest.json", "w"))
print("data ok")
