#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   arya-STARK - Installation Automatique          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}[1/6]${NC} Vérification des prérequis..."
python3 --version && echo -e "  ${GREEN}✓${NC} Python trouvé"
rustc --version && echo -e "  ${GREEN}✓${NC} Rust trouvé"

echo ""
echo -e "${YELLOW}[2/6]${NC} Installation Python..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -e .
echo -e "  ${GREEN}✓${NC} Package installé"

echo ""
echo -e "${YELLOW}[3/6]${NC} Compilation STARK prover..."
cd rust/stark-prover
cargo build --release --quiet
echo -e "  ${GREEN}✓${NC} STARK prover compilé"
cd ../..

echo ""
echo -e "${YELLOW}[4/6]${NC} Compilation ML-DSA bridge..."
export LIBCLANG_PATH=/usr/lib/llvm-15/lib
cd rust/mldsa-bridge
cargo build --release --quiet
echo -e "  ${GREEN}✓${NC} ML-DSA compilé"
cd ../..

echo ""
echo -e "${YELLOW}[5/6]${NC} Configuration environnement..."
cat > setup_env.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export ARYA_STARK_PROVE_BIN="$(pwd)/rust/stark-prover/target/release/prove"
export ARYA_STARK_VERIFY_BIN="$(pwd)/rust/stark-prover/target/release/verify"
export ARYA_STARK_KEYGEN_BIN="$(pwd)/rust/mldsa-bridge/target/release/keygen"
export ARYA_STARK_SIGN_BIN="$(pwd)/rust/mldsa-bridge/target/release/sign"
export ARYA_STARK_VERIFY_BIN_MLDSA="$(pwd)/rust/mldsa-bridge/target/release/verify_sig"
export ARYA_STARK_DISABLE_OQS=0
echo "✓ Environment activé"
EOF
chmod +x setup_env.sh
echo -e "  ${GREEN}✓${NC} setup_env.sh créé"

echo ""
echo -e "${YELLOW}[6/6]${NC} Tests..."
source setup_env.sh
pytest python/tests/ -q -m "not slow" || echo -e "${YELLOW}Certains tests ont échoué${NC}"

echo ""
echo -e "${GREEN}✓ Installation terminée !${NC}"
echo "Lancez: source setup_env.sh"
