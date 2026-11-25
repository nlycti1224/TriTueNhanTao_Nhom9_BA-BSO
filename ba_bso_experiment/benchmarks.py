import numpy as np

# Danh sách 30 seed cố định dùng chung cho BA và BSO (Bước 4)
SEEDS = list(range(1, 31))
DEFAULT_DIM = 30


def clip_bounds(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Sửa biên: ép mọi phần tử của vector nghiệm về trong đoạn [low, high]."""
    return np.clip(x, low, high)


# ===== 5 HÀM KIỂM THỬ CHUẨN (BƯỚC 2) =====

def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    return float(10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
    return float((term1 + 20.0) + (term2 + np.exp(1.0)))


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = x.size
    idx = np.arange(1, d + 1, dtype=float)
    sum_part = np.sum(x ** 2) / 4000.0
    prod_part = np.prod(np.cos(x / np.sqrt(idx)))
    return float(sum_part - prod_part + 1.0)


# Bảng cấu hình chuẩn dùng chung cho ba.py, bso.py, run_all.py
BENCHMARKS = {
    "Sphere":     {"func": sphere,     "bounds": (-100.0, 100.0),     "opt_val": 0.0},
    "Rosenbrock": {"func": rosenbrock, "bounds": (-2.048, 2.048),     "opt_val": 1.0},
    "Rastrigin":  {"func": rastrigin,  "bounds": (-5.12, 5.12),       "opt_val": 0.0},
    "Ackley":     {"func": ackley,     "bounds": (-32.768, 32.768),   "opt_val": 0.0},
    "Griewank":   {"func": griewank,   "bounds": (-600.0, 600.0),     "opt_val": 0.0},
}


# ===== UNIT-TEST KIỂM TRA (BƯỚC 3) =====
if __name__ == "__main__":
    print(f"Kiểm tra 5 hàm benchmark tại D = {DEFAULT_DIM}:")
    for name, cfg in BENCHMARKS.items():
        x_star = np.full(DEFAULT_DIM, cfg["opt_val"], dtype=float)
        val = cfg["func"](x_star)
        assert abs(val) < 1e-12, f"Lỗi ở hàm {name}: f(x*) = {val}"
        print(f"  [OK] {name:<11} | bounds={cfg['bounds']} | f(x*) = {val}")

    # Kiểm tra clip_bounds
    sample = np.array([-999.0, 0.0, 999.0])
    assert np.allclose(clip_bounds(sample, -10.0, 10.0), [-10.0, 0.0, 10.0])
    print(f"  [OK] clip_bounds & SEEDS ({len(SEEDS)} seeds: {SEEDS[0]}..{SEEDS[-1]})")
