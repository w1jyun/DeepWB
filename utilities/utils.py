# torch_utils_white_balance.py
import torch
import matplotlib.pyplot as plt

# ---------------------------
# Core tensor utilities
# ---------------------------

def out_of_gamut_clipping_t(I: torch.Tensor) -> torch.Tensor:
    """
    I: (..., C, H, W) 또는 (..., H, W, C) 가능, 값 범위 임의
    반환: [0,1]로 클램프된 텐서 (입력과 동일 shape)
    """
    return I.clamp_(0.0, 1.0) if I.is_floating_point() else I


def _to_N3(I: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """
    임의 shape의 이미지 텐서를 (N, 3) 로 펴는 헬퍼.
    허용 입력:
      - (H, W, 3)  또는 (3, H, W)
      - (B, 3, H, W) 또는 (B, H, W, 3)
    반환:
      - flat: (N, 3) float32
      - meta: (orig_shape, is_chw, batch_dims)
    """
    if I.ndim == 3:
        # (C,H,W) or (H,W,C)
        if I.shape[0] == 3:  # (3,H,W)
            C, H, W = I.shape
            flat = I.reshape(3, -1).T
            meta = ((C, H, W), True, 0)
        elif I.shape[-1] == 3:  # (H,W,3)
            H, W, C = I.shape
            flat = I.reshape(-1, 3)
            meta = ((H, W, C), False, 0)
        else:
            raise ValueError("3D tensor must be (3,H,W) or (H,W,3)")
    elif I.ndim == 4:
        # (B,3,H,W) or (B,H,W,3)
        if I.shape[1] == 3:  # (B,3,H,W)
            B, C, H, W = I.shape
            flat = I.permute(0, 2, 3, 1).reshape(-1, 3)
            meta = ((B, C, H, W), True, 1)
        elif I.shape[-1] == 3:  # (B,H,W,3)
            B, H, W, C = I.shape
            flat = I.reshape(-1, 3)
            meta = ((B, H, W, C), False, 1)
        else:
            raise ValueError("4D tensor must be (B,3,H,W) or (B,H,W,3)")
    else:
        raise ValueError("Tensor must be 3D or 4D image")
    return flat.to(dtype=torch.float32), meta


def _from_N3(X: torch.Tensor, meta: tuple) -> torch.Tensor:
    """
    (N,3) -> 원래 이미지 텐서 shape로 되돌림
    """
    orig_shape, is_chw, batch_dims = meta
    if len(orig_shape) == 3:  # 3D
        if is_chw:  # (3,H,W)
            _, H, W = orig_shape
            return X.T.reshape(3, H, W)
        else:       # (H,W,3)
            H, W, _ = orig_shape
            return X.reshape(H, W, 3)
    else:  # 4D
        if is_chw:  # (B,3,H,W)
            B, C, H, W = orig_shape
            return X.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B,3,H,W)
        else:       # (B,H,W,3)
            B, H, W, C = orig_shape
            return X.reshape(B, H, W, 3)


def kernelP_t(I_flat: torch.Tensor) -> torch.Tensor:
    """
    Polynomial kernel on torch:
      kernel(r,g,b) -> [r, g, b, rg, rb, gb, r^2, g^2, b^2, rgb, 1]
    입력: I_flat (N,3) in float
    반환: (N, 11)
    """
    r, g, b = I_flat[:, 0], I_flat[:, 1], I_flat[:, 2]
    ones = torch.ones_like(r)
    feats = torch.stack([
        r, g, b,
        r * g, r * b, g * b,
        r * r, g * g, b * b,
        r * g * b,
        ones
    ], dim=1)
    return feats


# ---------------------------
# Polynomial mapping (Torch)
# ---------------------------

class PolyColorMapper(torch.nn.Module):
    """
    image2 ≈ kernelP(image1) @ W
    - W shape: (11, 3)
    - 학습(추정)은 최소제곱 해 (closed-form / lstsq) 로 구함
    """
    def __init__(self, W: torch.Tensor):
        super().__init__()
        # register as parameter? 예측만 필요하면 buffer로 충분
        self.register_buffer('W', W)  # (11,3)

    @staticmethod
    def fit(image1: torch.Tensor, image2: torch.Tensor) -> "PolyColorMapper":
        """
        image1, image2: 같은 shape 이미지 텐서 (3D 혹은 4D), 값 범위 [0,1] 권장
        Torch 최소제곱으로 W 추정
        """
        I1_flat, meta1 = _to_N3(image1)
        I2_flat, meta2 = _to_N3(image2)
        if I1_flat.shape != I2_flat.shape:
            raise ValueError("image1 and image2 must have the same number of pixels")

        X = kernelP_t(I1_flat)          # (N,11)
        Y = I2_flat                     # (N,3)

        # lstsq: X @ W ≈ Y  ->  W = argmin ||XW - Y||
        # torch.linalg.lstsq returns solution for W in (11,3)
        W = torch.linalg.lstsq(X, Y).solution  # (11,3)
        return PolyColorMapper(W)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: 입력 이미지 (3D 또는 4D)
        반환: 매핑된 이미지 (입력과 동일한 shape)
        """
        flat, meta = _to_N3(image)
        out = kernelP_t(flat) @ self.W  # (N,3)
        return _from_N3(out, meta)


def get_mapping_func_t(image1: torch.Tensor, image2: torch.Tensor) -> PolyColorMapper:
    """ sklearn 없이 Torch만 사용해서 매핑 함수 추정 """
    return PolyColorMapper.fit(image1, image2)


def apply_mapping_func_t(image: torch.Tensor, mapper: PolyColorMapper) -> torch.Tensor:
    """ 추정된 torch 매핑 모듈을 적용 """
    return mapper(image)


# ---------------------------
# Color temperature interpolate (Torch)
# ---------------------------

def colorTempInterpolate_t(I_T: torch.Tensor, I_S: torch.Tensor):
    """
    텐서 간 보간으로 Fluorescent(F=3800K), Daylight(D=5500K), Cloudy(C=6500K) 생성
    입력: 동일 shape 텐서, 브로드캐스팅 가능
    반환: (I_F, I_D, I_C)
    """
    # CCT 값들 (float 스칼라)
    temps = {'T': 2850.0, 'F': 3800.0, 'D': 5500.0, 'C': 6500.0, 'S': 7500.0}
    cct1, cct2 = temps['T'], temps['S']
    inv = lambda x: 1.0 / x

    cct1inv, cct2inv = inv(cct1), inv(cct2)
    g_F = (inv(temps['F']) - cct2inv) / (cct1inv - cct2inv)
    g_D = (inv(temps['D']) - cct2inv) / (cct1inv - cct2inv)
    g_C = (inv(temps['C']) - cct2inv) / (cct1inv - cct2inv)

    I_F = g_F * I_T + (1 - g_F) * I_S
    I_D = g_D * I_T + (1 - g_D) * I_S
    I_C = g_C * I_T + (1 - g_C) * I_S
    return I_F, I_D, I_C


def colorTempInterpolate_w_target_t(I_T: torch.Tensor, I_S: torch.Tensor, target_temp: float) -> torch.Tensor:
    """
    임의 목표 색온도(target_temp, Kelvin)로 보간 값 반환
    """
    cct1, cct2 = 2850.0, 7500.0
    inv = lambda x: 1.0 / x
    g = (inv(target_temp) - inv(cct2)) / (inv(cct1) - inv(cct2))
    return g * I_T + (1 - g) * I_S


# ---------------------------
# I/O helpers (Torch-first)
# ---------------------------

def to_image_tensor_u8(I: torch.Tensor) -> torch.Tensor:
    """
    [0,1] float 이미지 텐서를 0~255 uint8 텐서로 변환 (시각화/저장용)
    입력: (H,W,3) / (3,H,W) / 배치 모두 허용
    반환: 입력과 동일한 layout, dtype=uint8 (CPU)
    """
    J = (I.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
    return J


# ---------------------------
# Visualization (matplotlib)
# ---------------------------

def _as_numpy_img(img: torch.Tensor):
    """
    텐서 -> numpy(H,W,3) 변환 (표시용)
    """
    if img.ndim == 3:
        if img.shape[0] == 3:    # (3,H,W)
            arr = img.detach().clamp(0,1).cpu().permute(1,2,0).numpy()
        elif img.shape[-1] == 3: # (H,W,3)
            arr = img.detach().clamp(0,1).cpu().numpy()
        else:
            raise ValueError("Expected 3 channels")
    else:
        raise ValueError("imshow helper expects 3D single image tensor")
    return arr


def imshow_t(img: torch.Tensor, *arguments: torch.Tensor, colortemp: int | None = None):
    """
    원본 imshow를 텐서 버전으로.
    - 입력 모두 torch 텐서(값 [0,1]) 가정, 내부에서 표시 직전에만 numpy로 변환
    """
    outimgs_num = len(arguments)

    if outimgs_num == 1 and not colortemp:
        titles = ["input", "awb"]
    elif outimgs_num == 1 and colortemp:
        titles = ["input", f"output ({int(colortemp)}K)"]
    elif outimgs_num == 5:
        titles = ["input", "tungsten", "fluorescent", "daylight", "cloudy", "shade"]
    elif outimgs_num == 6:
        titles = ["input", "awb", "tungsten", "fluorescent", "daylight", "cloudy", "shade"]
    else:
        raise Exception('Unexpected number of output images')

    if outimgs_num < 3:
        fig, ax = plt.subplots(1, outimgs_num + 1)
        ax = ax if isinstance(ax, (list, tuple)) else [ax]
        ax[0].set_title(titles[0])
        ax[0].imshow(_as_numpy_img(img))
        ax[0].axis('off')
        for i, image in enumerate(arguments, start=1):
            ax[i].set_title(titles[i])
            ax[i].imshow(_as_numpy_img(image))
            ax[i].axis('off')
    else:
        fig, ax = plt.subplots(2 + (outimgs_num + 1) % 3, 3)
        ax[0][0].set_title(titles[0])
        ax[0][0].imshow(_as_numpy_img(img))
        ax[0][0].axis('off')
        i = 1
        for image in arguments:
            if i == outimgs_num and outimgs_num == 6:
                ax[2][1].set_title(titles[i])
                ax[2][1].imshow(_as_numpy_img(image))
                ax[2][1].axis('off')
                ax[2][0].axis('off')
            else:
                ax[i // 3][i % 3].set_title(titles[i])
                ax[i // 3][i % 3].imshow(_as_numpy_img(image))
                ax[i // 3][i % 3].axis('off')
            i += 1

    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.show()
