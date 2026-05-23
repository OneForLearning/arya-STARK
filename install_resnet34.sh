#!/bin/bash

################################################################################
# arya-STARK ResNet-34 Installation Script
# Automated setup for MLP → ResNet-34 migration
# 
# Usage: bash install_resnet34.sh
#
# This script:
#   1. Copies ResNet-34 implementation
#   2. Updates config files
#   3. Verifies installation
#   4. Runs quick test
#   5. Reports status
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARYA_ROOT="${ARYA_ROOT:-.}"  # Default to current directory
MODELS_DIR="$ARYA_ROOT/python/arya_stark/models"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  arya-STARK ResNet-34 Installation Script              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# Step 1: Check Python version
################################################################################
echo -e "${YELLOW}[1/6]${NC} Checking Python version..."

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || [ "$MAJOR" -eq 3 -a "$MINOR" -lt 8 ]; then
    echo -e "${RED}✗ Python 3.8+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
echo ""

################################################################################
# Step 2: Check required packages
################################################################################
echo -e "${YELLOW}[2/6]${NC} Checking required packages..."

MISSING_PACKAGES=()

for package in numpy torch; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
        echo -e "${RED}  ✗ $package${NC} (missing)"
    else
        echo -e "${GREEN}  ✓ $package${NC}"
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Installing missing packages:${NC}"
    pip install numpy torch
fi
echo ""

################################################################################
# Step 3: Copy ResNet-34 implementation
################################################################################
echo -e "${YELLOW}[3/6]${NC} Installing ResNet-34 implementation..."

if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${RED}✗ Models directory not found: $MODELS_DIR${NC}"
    exit 1
fi

RESNET_SOURCE="$SCRIPT_DIR/resnet34_implementation.py"
RESNET_DEST="$MODELS_DIR/resnet34.py"

if [ ! -f "$RESNET_SOURCE" ]; then
    echo -e "${RED}✗ Source file not found: $RESNET_SOURCE${NC}"
    exit 1
fi

cp "$RESNET_SOURCE" "$RESNET_DEST"
echo -e "${GREEN}✓ Copied: $RESNET_DEST${NC}"
echo ""

################################################################################
# Step 4: Update models/__init__.py
################################################################################
echo -e "${YELLOW}[4/6]${NC} Updating models/__init__.py..."

INIT_FILE="$MODELS_DIR/__init__.py"

# Check if already updated
if grep -q "ResNet34Model" "$INIT_FILE"; then
    echo -e "${GREEN}✓ Already updated${NC}"
else
    # Add import
    if grep -q "from .mlp import MLPModel" "$INIT_FILE"; then
        sed -i 's/from \.mlp import MLPModel/from .mlp import MLPModel\nfrom .resnet34 import ResNet34Model/' "$INIT_FILE"
        echo -e "${GREEN}✓ Added import${NC}"
    else
        echo -e "${YELLOW}⚠ Manual update needed for imports${NC}"
    fi
    
    # Add to __all__
    if grep -q '__all__' "$INIT_FILE"; then
        sed -i 's/__all__ = \[\([^]]*\)\]/__all__ = [\1, "ResNet34Model"]/' "$INIT_FILE"
        echo -e "${GREEN}✓ Added to __all__${NC}"
    fi
fi
echo ""

################################################################################
# Step 5: Verify installation
################################################################################
echo -e "${YELLOW}[5/6]${NC} Verifying installation..."

# Test Python import
python3 << 'PYTHON_TEST'
import sys
try:
    from arya_stark.models import ResNet34Model
    print("✓ Import successful")
    
    # Test initialization
    model = ResNet34Model()
    print(f"✓ Model created: {model.total_params:,} parameters")
    
    # Verify parameter count
    if model.total_params == 21_797_122:
        print("✓ Parameter count verified (21,797,122)")
    else:
        print(f"✗ Unexpected parameter count: {model.total_params:,}")
        sys.exit(1)
    
    print("✓ All checks passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_TEST

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Verification successful${NC}"
else
    echo -e "${RED}✗ Verification failed${NC}"
    exit 1
fi
echo ""

################################################################################
# Step 6: Quick functionality test
################################################################################
echo -e "${YELLOW}[6/6]${NC} Running quick functionality test..."

python3 << 'PYTHON_TEST'
import numpy as np
from arya_stark.models import ResNet34Model

print("Testing ResNet-34 functionality...")

# Create model
model = ResNet34Model(seed=42)

# Test forward pass
X = np.random.randn(4, 256, 256, 3).astype(np.float32)
y = np.array([0, 1, 0, 1])

# Forward pass
logits = model.forward(X)
assert logits.shape == (4, 2), f"Wrong shape: {logits.shape}"
print("✓ Forward pass OK")

# Loss
loss = model.loss(X, y)
assert 0 <= loss < 100, f"Unrealistic loss: {loss}"
print(f"✓ Loss computation OK ({loss:.4f})")

# Accuracy
acc = model.accuracy(X, y)
assert 0 <= acc <= 1, f"Invalid accuracy: {acc}"
print(f"✓ Accuracy computation OK ({acc:.2%})")

# Gradient
grad = model.gradient(X, y)
assert grad.shape == (model.total_params,), f"Wrong gradient shape: {grad.shape}"
assert not np.allclose(grad, 0), "Zero gradient (suspicious)"
print(f"✓ Gradient computation OK (norm={np.linalg.norm(grad):.2e})")

# Parameter get/set
params_orig = model.get_flat_params()
params_new = params_orig + 0.1
model.set_flat_params(params_new)
params_retrieved = model.get_flat_params()
assert np.allclose(params_retrieved, params_new), "Parameter set/get mismatch"
print("✓ Parameter management OK")

print("\n✓ All functionality tests passed!")
PYTHON_TEST

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Functionality test passed${NC}"
else
    echo -e "${RED}✗ Functionality test failed${NC}"
    exit 1
fi
echo ""

################################################################################
# Success!
################################################################################
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ ResNet-34 Installation Complete!                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Summary:${NC}"
echo "  • ResNet-34 code: $RESNET_DEST"
echo "  • Parameters: 21,797,122"
echo "  • Status: Ready to use"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Read: arya-STARK_ResNet34_Quick_Integration.md"
echo "  2. Run:  python scripts/run_resnet34_eval.py"
echo "  3. Test: python -m pytest tests/test_resnet_basic.py -v"
echo ""

echo -e "${GREEN}Ready to train! 🚀${NC}"
echo ""

exit 0
