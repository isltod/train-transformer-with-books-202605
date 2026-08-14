#!/bin/bash
# =============================================================================
# WSL2 Ubuntu 22.04 — TensorFlow GPU 설치 및 검증 스크립트
# =============================================================================
# 사용법: 스크립트를 실행하려는 디렉토리로 이동 후 bash setup_tf_gpu.sh
# 전제조건: WSL2 Ubuntu 22.04 설치 완료, 호스트에 NVIDIA 드라이버 설치 완료
# =============================================================================

set -e  # 에러 발생 시 즉시 중단

VENV_DIR="$(pwd)/.venv"
PYTHON_VER="python3.10"

echo "=========================================="
echo " TensorFlow GPU 설치 스크립트"
echo "=========================================="
echo ""

# ----- 1단계: 시스템 패키지 설치 -----
echo "[1/5] python3-pip, python3-venv 설치 중..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv
echo "  ✓ 시스템 패키지 설치 완료"
echo ""

# ----- 2단계: 가상환경 생성 -----
echo "[2/5] 가상환경 생성 중: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "  ⚠ 기존 가상환경이 존재합니다. 삭제하고 다시 생성합니다."
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
echo "  ✓ 가상환경 생성 완료 (pip $(pip --version | awk '{print $2}'))"
echo ""

# ----- 3단계: TensorFlow GPU 설치 -----
echo "[3/5] TensorFlow GPU 설치 중 (CUDA/cuDNN 포함)..."
echo "  (수 분 소요될 수 있습니다)"
pip install "tensorflow[and-cuda]"
echo "  ✓ TensorFlow 설치 완료"
echo ""

# ----- 4단계: LD_LIBRARY_PATH 자동 설정 -----
echo "[4/5] 가상환경 activate에 CUDA 라이브러리 경로 등록 중..."
NVIDIA_PKG_DIR="$VENV_DIR/lib/$PYTHON_VER/site-packages/nvidia"

# 이미 추가되어 있는지 확인
if grep -q "NVIDIA CUDA libs for TensorFlow" "$VENV_DIR/bin/activate" 2>/dev/null; then
    echo "  ⚠ 이미 등록되어 있습니다. 건너뜁니다."
else
    cat >> "$VENV_DIR/bin/activate" << 'EOF'

# NVIDIA CUDA libs for TensorFlow
VENV_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVIDIA_DIR=$(find "$VENV_BASE_DIR/lib" -maxdepth 2 -name "site-packages" 2>/dev/null)/nvidia
if [ -d "$NVIDIA_DIR" ]; then
    export LD_LIBRARY_PATH=$(find "$NVIDIA_DIR" -name lib -type d 2>/dev/null | tr '\n' ':'):$LD_LIBRARY_PATH
fi
EOF
    echo "  ✓ LD_LIBRARY_PATH 자동 설정 등록 완료"
fi

# 현재 셸에도 적용
export LD_LIBRARY_PATH=$(find "$NVIDIA_PKG_DIR" -name lib -type d | tr '\n' ':'):$LD_LIBRARY_PATH
echo ""

# ----- 5단계: GPU 인식 및 연산 테스트 -----
echo "[5/5] TensorFlow GPU 테스트 실행 중..."
echo ""

python3 << 'PYEOF'
import tensorflow as tf

print(f"  TensorFlow 버전: {tf.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"  인식된 GPU 수: {len(gpus)}")

if not gpus:
    print("")
    print("  ❌ GPU를 인식하지 못했습니다!")
    print("     - 호스트 Windows에 NVIDIA 드라이버가 설치되어 있는지 확인하세요")
    print("     - nvidia-smi 명령어가 WSL2에서 동작하는지 확인하세요")
    exit(1)

print("")
for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    name = details.get("device_name", "Unknown")
    cc = details.get("compute_capability", (0, 0))
    print(f"  ✓ {gpu.name}: {name} (compute {cc[0]}.{cc[1]})")

print("")
print("  GPU 연산 테스트...")
with tf.device("/GPU:0"):
    a = tf.random.normal([2000, 2000])
    b = tf.random.normal([2000, 2000])
    c = tf.matmul(a, b)
    print(f"  ✓ 행렬곱 테스트 통과 (shape: {c.shape})")

print("")
print("=" * 42)
print("  ✅ TensorFlow GPU 설치 및 검증 완료!")
print("=" * 42)
print("")
print("  사용법:")
print("    source .venv/bin/activate")
print("    python3 your_script.py")
print("")
PYEOF
