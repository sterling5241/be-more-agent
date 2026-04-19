"""
discord_bot.py  —  Be More Agent, Discord edition
Drop into the root of the be-more-agent repo alongside agent.py.

Extra pip installs (add to requirements.txt):
    discord.py[voice]>=2.3.2
    PyNaCl
    aiohttp
    duckduckgo-search

Run:
    export DISCORD_TOKEN=your_bot_token_here
    source venv/bin/activate
    python discord_bot.py

Bot permissions needed (in Discord Developer Portal):
    - Send Messages
    - Embed Links
    - Attach Files
    - Read Message History
    - Connect (voice)
    - Speak (voice)
    - Use Voice Activity
    - Manage Messages (to pin the face embed)
    Privileged intents: Message Content Intent
"""

import asyncio
import io
import json
import os
import random
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import aiohttp
import discord
from discord.ext import commands, tasks

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_FILE = "config.json"
MEMORY_FILE = "discord_memory.json"
FACES_DIR   = Path("faces")
PIPER_DIR   = Path("piper")
WHISPER_BIN = Path("whisper.cpp/main")   # adjust if your whisper build is elsewhere

DEFAULT_CONFIG = {
    "text_model":           "gemma3:1b",
    "vision_model":         "moondream",
    "voice_model":          "piper/en_GB-semaine-medium.onnx",
    "chat_memory":          True,
    "system_prompt_extras": "You are a helpful robot assistant. Keep responses short and cute.",
}

def load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    except Exception:
        return dict(DEFAULT_CONFIG)

cfg = load_config()

# ──────────────────────────────────────────────────────────────────────────────
# State machine
# ──────────────────────────────────────────────────────────────────────────────
# States: idle | listening | thinking | speaking | error
# Face embed updates on EVERY state change.
# Avatar swaps only on major transitions: idle <-> active (rate limit ~2/hr).

STATES      = ["idle", "listening", "thinking", "speaking", "error"]
ACTIVE_STATES = {"listening", "thinking", "speaking"}

class AgentState:
    def __init__(self):
        self.current  = "idle"
        self._prev    = "idle"

    def set(self, new_state: str):
        self._prev   = self.current
        self.current = new_state

    def crossed_idle_boundary(self) -> bool:
        """True when we just went idle->active or active->idle."""
        was_idle = self._prev not in ACTIVE_STATES
        is_idle  = self.current not in ACTIVE_STATES
        return was_idle != is_idle

agent_state = AgentState()

# ──────────────────────────────────────────────────────────────────────────────
# Face helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_face_image(state: str) -> Path | None:
    """Pick a random PNG from faces/<state>/. Falls back to idle."""
    folder = FACES_DIR / state
    if not folder.exists():
        folder = FACES_DIR / "idle"
    pngs = list(folder.glob("*.png"))
    return random.choice(pngs) if pngs else None


def state_colour(state: str) -> discord.Colour:
    return {
        "idle":      discord.Colour.greyple(),
        "listening": discord.Colour.green(),
        "thinking":  discord.Colour.gold(),
        "speaking":  discord.Colour.blurple(),
        "error":     discord.Colour.red(),
    }.get(state, discord.Colour.default())


def state_label(state: str) -> str:
    return {
        "idle":      "💤  Idle",
        "listening": "👂  Listening...",
        "thinking":  "🤔  Thinking...",
        "speaking":  "💬  Speaking",
        "error":     "❌  Error",
    }.get(state, state.title())

# ──────────────────────────────────────────────────────────────────────────────
# Face embed manager (text channel only)
# ──────────────────────────────────────────────────────────────────────────────

class FaceManager:
    """
    Keeps track of the pinned face embed message in each text channel.
    Call update(channel, state) after every state change.
    """
    def __init__(self):
        # channel_id -> Message
        self._pinned: dict[int, discord.Message] = {}
        # avatar rate-limit: only swap when crossing idle boundary
        self._last_avatar_swap = 0.0

    async def update(self, channel: discord.TextChannel, new_state: str):
        agent_state.set(new_state)
        img_path = get_face_image(new_state)

        embed = discord.Embed(
            title=state_label(new_state),
            colour=state_colour(new_state),
        )
        embed.set_footer(text="Be More Agent")

        if img_path:
            fname = f"face_{new_state}.png"
            file  = discord.File(str(img_path), filename=fname)
            embed.set_image(url=f"attachment://{fname}")
        else:
            file = None

        existing = self._pinned.get(channel.id)
        try:
            if existing:
                # Edit in-place (no file re-send possible on edit, so delete+resend)
                await existing.delete()
            kwargs = {"embed": embed}
            if file:
                kwargs["file"] = file
            msg = await channel.send(**kwargs)
            try:
                await msg.pin()
            except discord.Forbidden:
                pass  # no pin permission — that's fine
            self._pinned[channel.id] = msg
        except Exception as e:
            print(f"[FaceManager] Could not update embed: {e}")

        # Avatar swap — only on idle <-> active boundary, max ~once per 30 min
        if agent_state.crossed_idle_boundary():
            now = time.time()
            if now - self._last_avatar_swap > 1800:
                await self._swap_avatar(channel.guild.me, new_state)
                self._last_avatar_swap = now

    async def _swap_avatar(self, me: discord.Member, state: str):
        img_path = get_face_image(state)
        if not img_path:
            return
        try:
            with open(img_path, "rb") as f:
                await me.guild.me.edit(avatar=f.read())   # type: ignore
        except Exception as e:
            print(f"[FaceManager] Avatar swap failed: {e}")

face_mgr = FaceManager()

# ──────────────────────────────────────────────────────────────────────────────
# Chat memory (per-channel, separate file from Pi's chat_memory.json)
# ──────────────────────────────────────────────────────────────────────────────

def load_memory(channel_id: int) -> list:
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        return data.get(str(channel_id), [])
    except Exception:
        return []

def save_memory(channel_id: int, messages: list):
    data: dict = {}
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
    except Exception:
        pass
    data[str(channel_id)] = messages[-30:]
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# Web search (DuckDuckGo — same as agent.py)
# ──────────────────────────────────────────────────────────────────────────────

async def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return " | ".join(r["body"] for r in results) if results else ""
    except Exception:
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# Ollama LLM
# ──────────────────────────────────────────────────────────────────────────────

async def ask_ollama(user_input: str, history: list) -> str:
    search_ctx = await web_search(user_input)
    context    = f"\nWeb search context: {search_ctx}" if search_ctx else ""

    messages = [
        {"role": "system", "content": cfg["system_prompt_extras"] + context},
        *history,
        {"role": "user", "content": user_input},
    ]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/chat",
                json={"model": cfg["text_model"], "messages": messages, "stream": False},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.json()
                return data["message"]["content"]
    except Exception as e:
        return f"(Ollama error: {e})"

# ──────────────────────────────────────────────────────────────────────────────
# Piper TTS  →  returns path to a .wav tempfile (caller must delete it)
# ──────────────────────────────────────────────────────────────────────────────

def tts_to_wav(text: str) -> Path | None:
    voice_model = cfg.get("voice_model", "")
    piper_exe   = PIPER_DIR / "piper"

    if not piper_exe.exists() or not Path(voice_model).exists():
        print("[TTS] Piper or voice model not found — skipping TTS")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [str(piper_exe), "--model", voice_model, "--output_file", str(out_path)],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[TTS] Piper error: {result.stderr.decode()}")
            out_path.unlink(missing_ok=True)
            return None
        return out_path
    except Exception as e:
        print(f"[TTS] Exception: {e}")
        out_path.unlink(missing_ok=True)
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Whisper STT  —  records from a discord voice sink, returns transcript
# ──────────────────────────────────────────────────────────────────────────────

class WaveSink(discord.sinks.WaveSink):
    """Thin wrapper so we can signal when recording is done."""
    pass

async def record_and_transcribe(vc: discord.VoiceClient, seconds: int = 5) -> str:
    """Record `seconds` of audio from VC then run Whisper."""
    sink = WaveSink()
    vc.start_recording(sink, lambda *_: None)
    await asyncio.sleep(seconds)
    vc.stop_recording()

    # Merge all users into one wav
    all_frames = b""
    sample_rate = 48000
    channels    = 2

    for user_id, audio in sink.audio_data.items():
        audio.file.seek(0)
        all_frames += audio.file.read()

    if not all_frames:
        return ""

    # Write merged wav
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp_wav.name, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(all_frames)
    tmp_wav.close()

    # Run Whisper
    if not WHISPER_BIN.exists():
        print("[STT] whisper.cpp binary not found")
        Path(tmp_wav.name).unlink(missing_ok=True)
        return ""

    model_path = Path("whisper.cpp/models/ggml-base.en.bin")
    if not model_path.exists():
        # try to find any model
        models = list(Path("whisper.cpp/models").glob("*.bin"))
        if not models:
            print("[STT] No Whisper model found")
            Path(tmp_wav.name).unlink(missing_ok=True)
            return ""
        model_path = models[0]

    try:
        result = subprocess.run(
            [str(WHISPER_BIN), "-m", str(model_path), "-f", tmp_wav.name, "--no-timestamps", "-otxt"],
            capture_output=True, timeout=30
        )
        txt_file = Path(tmp_wav.name + ".txt")
        if txt_file.exists():
            transcript = txt_file.read_text().strip()
            txt_file.unlink(missing_ok=True)
        else:
            transcript = result.stdout.decode().strip()
    except Exception as e:
        print(f"[STT] Whisper error: {e}")
        transcript = ""
    finally:
        Path(tmp_wav.name).unlink(missing_ok=True)

    return transcript

# ──────────────────────────────────────────────────────────────────────────────
# Bot setup
# ──────────────────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states    = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Track which text channel to update faces in, per guild
face_channels: dict[int, discord.TextChannel] = {}

# ──────────────────────────────────────────────────────────────────────────────
# Shared pipeline: think + reply
# Used by both !ask (text) and voice listener
# ──────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    user_input: str,
    channel_id: int,
    face_channel: discord.TextChannel | None,
    vc: discord.VoiceClient | None = None,
) -> str:
    """
    1. Set state → thinking, update face embed (if face_channel given)
    2. Ask Ollama
    3. Set state → speaking
    4. If vc given: play TTS audio in voice channel
    5. Set state → idle, update face embed
    Returns the text reply.
    """

    if face_channel:
        await face_mgr.update(face_channel, "thinking")

    history = load_memory(channel_id) if cfg["chat_memory"] else []
    reply   = await ask_ollama(user_input, history)

    if cfg["chat_memory"]:
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})
        save_memory(channel_id, history)

    if vc and vc.is_connected():
        if face_channel:
            await face_mgr.update(face_channel, "speaking")
        wav_path = await asyncio.get_event_loop().run_in_executor(None, tts_to_wav, reply)
        if wav_path:
            source = discord.FFmpegPCMAudio(str(wav_path))
            vc.play(source)
            while vc.is_playing():
                await asyncio.sleep(0.5)
            wav_path.unlink(missing_ok=True)

    if face_channel:
        await face_mgr.update(face_channel, "idle")

    return reply

# ──────────────────────────────────────────────────────────────────────────────
# Text commands
# ──────────────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    print(f"[Bot] Online as {bot.user} (id: {bot.user.id})")
    print(f"[Bot] Using model: {cfg['text_model']}")


@bot.command(name="ask", help="Ask the bot anything. Usage: !ask <question>")
async def cmd_ask(ctx: commands.Context, *, question: str):
    """Main text command — updates face embed, replies in channel."""
    face_channel = face_channels.get(ctx.guild.id, ctx.channel)
    await face_mgr.update(face_channel, "listening")

    async with ctx.typing():
        reply = await run_pipeline(
            user_input   = question,
            channel_id   = ctx.channel.id,
            face_channel = face_channel,
        )

    # Split replies > 2000 chars (Discord limit)
    for chunk in [reply[i:i+2000] for i in range(0, len(reply), 2000)]:
        await ctx.reply(chunk)


@bot.command(name="facechannel", help="Set this channel as the face display channel.")
async def cmd_facechannel(ctx: commands.Context):
    face_channels[ctx.guild.id] = ctx.channel
    await ctx.send(f"Face channel set to {ctx.channel.mention}.")
    await face_mgr.update(ctx.channel, "idle")


@bot.command(name="join", help="Join your voice channel.")
async def cmd_join(ctx: commands.Context):
    if not ctx.author.voice:
        await ctx.send("You're not in a voice channel!")
        return
    vc = await ctx.author.voice.channel.connect()
    await ctx.send(f"Joined **{vc.channel.name}**. Use `!listen` to start listening.")


@bot.command(name="leave", help="Leave the voice channel.")
async def cmd_leave(ctx: commands.Context):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I'm not in a voice channel.")


@bot.command(name="listen", help="Record 5 seconds of audio, transcribe, and reply in VC.")
async def cmd_listen(ctx: commands.Context):
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        await ctx.send("I'm not in a voice channel. Use `!join` first.")
        return

    face_channel = face_channels.get(ctx.guild.id)

    if face_channel:
        await face_mgr.update(face_channel, "listening")

    status_msg = await ctx.send("🎙️ Listening for 5 seconds...")

    transcript = await record_and_transcribe(vc, seconds=5)
    await status_msg.delete()

    if not transcript:
        if face_channel:
            await face_mgr.update(face_channel, "error")
            await asyncio.sleep(2)
            await face_mgr.update(face_channel, "idle")
        await ctx.send("I couldn't hear anything.")
        return

    await ctx.send(f"**Heard:** {transcript}")

    reply = await run_pipeline(
        user_input   = transcript,
        channel_id   = ctx.channel.id,
        face_channel = face_channel,
        vc           = vc,
    )
    await ctx.send(f"**Reply:** {reply[:2000]}")


@bot.command(name="clearmemory", help="Wipe conversation memory for this channel.")
async def cmd_clearmemory(ctx: commands.Context):
    save_memory(ctx.channel.id, [])
    await ctx.reply("Memory cleared for this channel. 🧹")


@bot.command(name="status", help="Show current bot state.")
async def cmd_status(ctx: commands.Context):
    face_channel = face_channels.get(ctx.guild.id)
    fc_mention   = face_channel.mention if face_channel else "not set — use `!facechannel`"
    await ctx.send(
        f"**State:** {agent_state.current}\n"
        f"**Model:** {cfg['text_model']}\n"
        f"**Face channel:** {fc_mention}\n"
        f"**Memory:** {'on' if cfg['chat_memory'] else 'off'}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Handle @mentions as !ask
# ──────────────────────────────────────────────────────────────────────────────

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if bot.user and bot.user.mentioned_in(message):
        question = message.content.replace(f"<@{bot.user.id}>", "").strip()
        if question:
            ctx = await bot.get_context(message)
            await cmd_ask(ctx, question=question)
            return
    await bot.process_commands(message)

# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

TOKEN = os.environ.get("DISCORD_TOKEN", "")
if not TOKEN:
    raise RuntimeError("Set the DISCORD_TOKEN environment variable before running.")

bot.run(TOKEN)
