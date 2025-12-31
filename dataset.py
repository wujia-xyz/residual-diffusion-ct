import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch_radon import RadonFanbeam

angles = torch.linspace(0, 2 * torch.pi, 64)
resolution = 512
radon = RadonFanbeam(
    resolution,
    angles,
    source_distance=590.0,
    det_distance=490.0,
    det_count=736,
    det_spacing=2.1,
    clip_to_circle=False
)


class IMADataset(Dataset):
    def __init__(self, folder, exts=['IMA'], device='cuda'):
        super().__init__()
        self.folder = folder
        self.device = device

        self.paths = []
        for subdir in Path(folder).iterdir():
            if subdir.is_dir():
                full_3mm_path = subdir / 'full_3mm'
                if full_3mm_path.is_dir():
                    self.paths.extend([p for ext in exts for p in full_3mm_path.glob(f'**/*.{ext}')])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        dicom_image = pydicom.dcmread(path)
        img_array = dicom_image.pixel_array.astype(np.float32)

        img_array[img_array > 2500] = 0
        img_array = img_array / 2500.0

        x_0 = torch.from_numpy(img_array).unsqueeze(0).to(self.device)

        sinogram = radon.forward(x_0)
        filtered_sinogram = radon.filter_sinogram(sinogram)
        fbp = radon.backward(filtered_sinogram)

        return x_0, fbp


class DicomDataset(Dataset):
    def __init__(self, folder, exts=['dcm'], device='cuda'):
        self.folder = Path(folder)
        self.device = device

        self.paths = []
        for ext in exts:
            files = list(self.folder.glob(f'*.{ext}'))
            self.paths.extend(files)
            print(f"Found {len(files)} files with extension .{ext} in {self.folder}")

        print(f"Total files found: {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        dicom_image = pydicom.dcmread(path)
        img_array = dicom_image.pixel_array.astype(np.float32)

        img_array[img_array > 2500] = 0
        img_array = img_array / 2500.0

        x_0 = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(self.device)

        sinogram = radon.forward(x_0)
        filtered_sinogram = radon.filter_sinogram(sinogram)
        fbp = radon.backward(filtered_sinogram)
        x_0 = x_0.squeeze(0)
        fbp = fbp.squeeze(0)

        return x_0, fbp
