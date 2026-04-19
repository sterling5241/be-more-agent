#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Be More Agent — Setup Script
# Supports: Raspberry Pi (aarch64) and x86_64 Linux (PC/server)
# ─────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🤖 Be More Agent Setup Script${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. System dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/7] Installing system tools (apt)...${NC}"
sudo apt update
sudo apt install -y \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    python-is-python3 \
    python3-tk \
    libasound2-dev \
    portaudio19-dev \
    liblapack-dev \
    libblas-dev \
    cmake \
    build-essential \
    espeak-ng \
    git \
    ffmpeg \
    wget \
    curl

# ─────────────────────────────────────────────────────────────────────────────
# 2. Create folders
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/7] Creating folders...${NC}"
mkdir -p piper
mkdir -p voices
mkdir -p sounds/greeting_sounds
mkdir -p sounds/thinking_sounds
mkdir -p sounds/ack_sounds
mkdir -p sounds/error_sounds
mkdir -p faces/idle
mkdir -p faces/listening
mkdir -p faces/thinking
mkdir -p faces/speaking
mkdir -p faces/error
mkdir -p faces/warmup

# ─────────────────────────────────────────────────────────────────────────────
# 3. Piper TTS binary (architecture-aware)
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/7] Setting up Piper TTS binary...${NC}"
ARCH=$(uname -m)
PIPER_VERSION="2023.11.14-2"

if [ "$ARCH" == "aarch64" ]; then
    echo -e "  Detected: Raspberry Pi (aarch64)"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"
elif [ "$ARCH" == "x86_64" ]; then
    echo -e "  Detected: x86_64 (PC/server)"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_x86_64.tar.gz"
elif [ "$ARCH" == "armv7l" ]; then
    echo -e "  Detected: armv7 (older Pi)"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_armv7l.tar.gz"
else
    echo -e "${RED}  ⚠️  Unknown architecture: $ARCH — skipping Piper download.${NC}"
    PIPER_URL=""
fi

if [ -n "$PIPER_URL" ]; then
    wget -O piper.tar.gz "$PIPER_URL"
    tar -xvf piper.tar.gz -C piper --strip-components=1
    rm piper.tar.gz
    echo -e "${GREEN}  ✅ Piper installed${NC}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Voice models
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/7] Downloading voice models...${NC}"

cd piper
wget -nc -O en_GB-semaine-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx
wget -nc -O en_GB-semaine-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json
cd ..

echo -e "  Downloading custom BMO voice..."
curl -L -o voices/bmo-custom.onnx \
    "https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx" || \
    echo -e "${YELLOW}  ⚠️  BMO voice not found at release URL — skipping.${NC}"
curl -L -o voices/bmo-custom.onnx.json \
    "https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx.json" || true

# ─────────────────────────────────────────────────────────────────────────────
# 5. Python virtual environment + dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[5/7] Setting up Python virtual environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip

# Force rebuild sounddevice against the system PortAudio
pip install --force-reinstall --no-cache-dir sounddevice

# Core agent dependencies
pip install -r requirements.txt

# Discord bot dependencies (only installs if not already present)
echo -e "  Installing Discord bot dependencies..."
pip install "discord.py[voice]" PyNaCl aiohttp scipy

echo -e "${GREEN}  ✅ Python environment ready${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# 6. AI models via Ollama
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[6/7] Pulling AI models...${NC}"

if command -v ollama &> /dev/null; then
    ollama pull gemma3:1b
    ollama pull moondream
    echo -e "${GREEN}  ✅ Ollama models ready${NC}"
else
    echo -e "${RED}  ❌ Ollama not found. Install it with:${NC}"
    echo -e "     curl -fsSL https://ollama.com/install.sh | sh"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. Wake word model
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[7/7] Checking wake word model...${NC}"

if [ ! -f "wakeword.onnx" ]; then
    echo -e "  Downloading default 'Hey Jarvis' wake word..."
    curl -L -o wakeword.onnx \
        https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx
    echo -e "${GREEN}  ✅ Wake word model downloaded${NC}"
else
    echo -e "  wakeword.onnx already exists — skipping."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}✨ Setup complete!${NC}"
echo ""
echo -e "  Run the original agent (Pi/local):"
echo -e "    ${YELLOW}source venv/bin/activate && python agent.py${NC}"
echo ""
echo -e "  Run the Discord bot:"
echo -e "    ${YELLOW}export DISCORD_TOKEN=your_token_here${NC}"
echo -e "    ${YELLOW}source venv/bin/activate && python discord_bot.py${NC}"
echo ""
