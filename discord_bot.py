import asyncio
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
import numpy as np
from discord.ext import commands
from openwakeword.model import Model as WakeWordModel
from scipy.signal import resample_poly
 
# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
 
CONFIG_FILE    = "config.json"
MEMORY_FILE    = "discord_memory.json"
FACES_DIR      = Path("faces")
PIPER_DIR      = Path("piper")
WHISPER_BIN    = Path("whisper.cpp/main")
WAKEWORD_MODEL = Path("wakeword.onnx")
 
# Discord voice output: 48 kHz stereo 16-bit PCM
# OpenWakeWord needs: 16 kHz mono 16-bit PCM, 80 ms frames = 1280 samples
DISCORD_SAMPLE_RATE = 48000
OWW_SAMPLE_RATE     = 16000
OWW_FRAME_SAMPLES   = 1280      # 80 ms at 16 kHz
SILENCE_FRAMES      = 20        # ~1.6 s of silence ends recording
MIN_RECORD_FRAMES   = 12        # ~1 s minimum speech before silence check
WAKE_THRESHOLD      = 0.5       # OWW confidence threshold
 
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
 
ACTIVE_STATES = {"listening", "thinking", "speaking"}
 
class AgentState:
    def __init__(self):
        self.current = "idle"
        self._prev   = "idle"
 
    def set(self, new_state: str):
        self._prev   = self.current
        self.current = new_state
 
    def crossed_idle_boundary(self) -> bool:
        """True when transitioning between idle and active (or vice versa)."""
        was_idle = self._prev   not in ACTIVE_STATES
        is_idle  = self.current not in ACTIVE_STATES
        return was_idle != is_idle
 
agent_state = AgentState()
 
# ──────────────────────────────────────────────────────────────────────────────
# Wake word model — loaded once at startup
# ──────────────────────────────────────────────────────────────────────────────
 
oww_model: WakeWordModel | None = None
 
def load_wakeword_model():
    global oww_model
    if not WAKEWORD_MODEL.exists():
        print(f"[WakeWord] {WAKEWORD_MODEL} not found — wake word disabled in VC")
        return
    try:
        oww_model = WakeWordModel(
            wakeword_models=[str(WAKEWORD_MODEL)],
            inference_framework="onnx",
        )
        print(f"[WakeWord] Loaded {WAKEWORD_MODEL}")
    except Exception as e:
        print(f"[WakeWord] Failed to load: {e}")
 
# ──────────────────────────────────────────────────────────────────────────────
# Audio conversion: Discord PCM → OWW frame
# ──────────────────────────────────────────────────────────────────────────────
 
def discord_pcm_to_mono16k(raw: bytes) -> np.ndarray:
    """
    Raw Discord PCM is 48 kHz, 16-bit, stereo.
    OWW needs 16 kHz, 16-bit, mono.
    Steps: parse → mix to mono → resample 48k→16k (ratio 1:3).
    """
    samples  = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    mono     = samples.mean(axis=1).astype(np.int16)
    # resample_poly(up=1, down=3) gives exact 16000/48000 = 1/3 ratio
    resampled = resample_poly(mono, up=1, down=3).astype(np.int16)
    return resampled
 
 
def is_silent(chunk: np.ndarray, threshold: int = 300) -> bool:
    """Energy-based VAD matching agent.py approach."""
    return int(np.abs(chunk).mean()) < threshold
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Discord raw audio sink → asyncio queue
# ──────────────────────────────────────────────────────────────────────────────
 
class QueueSink(discord.sinks.RawSink):
    """Pushes raw PCM from all VC users into an asyncio.Queue."""
 
    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue
 
    def write(self, data: bytes, user):
        try:
            self.queue.put_nowait(data)
        except asyncio.QueueFull:
            pass  # drop frame if consumer is behind
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Always-on voice listener loop
# ──────────────────────────────────────────────────────────────────────────────
 
async def voice_listener_loop(
    vc:           discord.VoiceClient,
    face_channel: discord.TextChannel | None,
    guild_id:     int,
):
    """
    Runs as a background task for as long as the bot is in a voice channel.
 
    Loop:
      1. Stream audio → OWW → wait for wake word
      2. Wake word detected → record until silence (VAD)
      3. Whisper STT on recorded audio
      4. Ollama LLM → reply text
      5. Piper TTS → play reply in VC
      6. Go back to step 1
    """
    if oww_model is None:
        print("[VoiceLoop] No wake word model loaded — loop aborted")
        return
 
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=300)
    sink = QueueSink(audio_queue)
    vc.start_recording(sink, lambda *_: None)
    print(f"[VoiceLoop] Listening for wake word in guild {guild_id}")
 
    # Accumulator for building complete OWW frames
    oww_buf = np.array([], dtype=np.int16)
 
    try:
        while vc.is_connected():
 
            # ── Phase 1: idle — feed OWW until wake word ──────────────────
            if face_channel:
                await face_mgr.update(face_channel, "idle")
 
            detected = False
            while vc.is_connected() and not detected:
                try:
                    raw = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
 
                chunk   = discord_pcm_to_mono16k(raw)
                oww_buf = np.concatenate([oww_buf, chunk])
 
                # Feed complete 80 ms frames to OWW
                while len(oww_buf) >= OWW_FRAME_SAMPLES:
                    frame   = oww_buf[:OWW_FRAME_SAMPLES]
                    oww_buf = oww_buf[OWW_FRAME_SAMPLES:]
                    preds   = oww_model.predict(frame)
                    if any(score >= WAKE_THRESHOLD for score in preds.values()):
                        detected = True
                        oww_buf  = np.array([], dtype=np.int16)  # reset buffer
                        break
 
            if not vc.is_connected():
                break
 
            print("[VoiceLoop] 🔔 Wake word detected")
            if face_channel:
                await face_mgr.update(face_channel, "listening")
 
            # Flush stale audio that arrived during wake word detection
            while not audio_queue.empty():
                audio_queue.get_nowait()
 
            # ── Phase 2: record speech until silence ──────────────────────
            speech_chunks: list[np.ndarray] = []
            silent_count  = 0
            frame_count   = 0
 
            while vc.is_connected():
                try:
                    raw = await asyncio.wait_for(audio_queue.get(), timeout=3.0)
                except asyncio.TimeoutError:
                    # Timeout = definitely done speaking
                    break
 
                chunk = discord_pcm_to_mono16k(raw)
                speech_chunks.append(chunk)
                frame_count += 1
 
                if frame_count >= MIN_RECORD_FRAMES:
                    if is_silent(chunk):
                        silent_count += 1
                        if silent_count >= SILENCE_FRAMES:
                            break
                    else:
                        silent_count = 0
 
            if not speech_chunks:
                continue
 
            # ── Phase 3: Whisper STT ──────────────────────────────────────
            audio_data = np.concatenate(speech_chunks)
            transcript = await asyncio.get_event_loop().run_in_executor(
                None, transcribe_audio, audio_data
            )
            transcript = transcript.strip()
 
            if not transcript:
                print("[VoiceLoop] Empty transcript — back to idle")
                continue
 
            print(f"[VoiceLoop] Heard: {transcript}")
            if face_channel:
                await face_channel.send(
                    f"🎙️ **Heard:** {transcript}", delete_after=60
                )
 
            # ── Phase 4 + 5: LLM pipeline + TTS playback ──────────────────
            reply = await run_pipeline(
                user_input   = transcript,
                channel_id   = face_channel.id if face_channel else guild_id,
                face_channel = face_channel,
                vc           = vc,
            )
 
            # Post reply text to the face channel too
            if face_channel:
                for chunk in [reply[i:i+2000] for i in range(0, len(reply), 2000)]:
                    await face_channel.send(f"🤖 {chunk}", delete_after=120)
 
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[VoiceLoop] Unexpected error: {e}")
    finally:
        try:
            vc.stop_recording()
        except Exception:
            pass
        print(f"[VoiceLoop] Stopped in guild {guild_id}")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Whisper STT  (blocking — run in executor)
# ──────────────────────────────────────────────────────────────────────────────
 
def transcribe_audio(audio: np.ndarray) -> str:
    """Write 16 kHz mono array to a temp wav, run whisper.cpp, return text."""
    if not WHISPER_BIN.exists():
        print("[STT] whisper.cpp binary not found")
        return ""
 
    models = list(Path("whisper.cpp/models").glob("*.bin"))
    if not models:
        print("[STT] No model in whisper.cpp/models/")
        return ""
    # Prefer base model; fall back to whatever is there
    model_path = next((m for m in models if "base" in m.name), models[0])
 
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav_path = Path(tmp.name)
 
    try:
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(OWW_SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
 
        result = subprocess.run(
            [
                str(WHISPER_BIN),
                "-m", str(model_path),
                "-f", str(wav_path),
                "--no-timestamps",
                "-otxt",
            ],
            capture_output=True,
            timeout=30,
        )
        txt_file = Path(str(wav_path) + ".txt")
        if txt_file.exists():
            text = txt_file.read_text().strip()
            txt_file.unlink(missing_ok=True)
        else:
            text = result.stdout.decode().strip()
        return text
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""
    finally:
        wav_path.unlink(missing_ok=True)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Face embed manager  (text channels only)
# ──────────────────────────────────────────────────────────────────────────────
 
def get_face_image(state: str) -> Path | None:
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
 
 
class FaceManager:
    def __init__(self):
        self._pinned: dict[int, discord.Message] = {}
        self._last_avatar_swap = 0.0
 
    async def update(self, channel: discord.TextChannel, new_state: str):
        agent_state.set(new_state)
        img_path = get_face_image(new_state)
 
        embed = discord.Embed(
            title=state_label(new_state),
            colour=state_colour(new_state),
        )
        embed.set_footer(text="Be More Agent")
 
        file = None
        if img_path:
            fname = f"face_{new_state}.png"
            file  = discord.File(str(img_path), filename=fname)
            embed.set_image(url=f"attachment://{fname}")
 
        # Delete old pinned message and resend (edit can't swap attached image)
        existing = self._pinned.get(channel.id)
        try:
            if existing:
                try:
                    await existing.delete()
                except Exception:
                    pass
            kwargs = {"embed": embed}
            if file:
                kwargs["file"] = file
            msg = await channel.send(**kwargs)
            try:
                await msg.pin()
            except discord.Forbidden:
                pass  # no pin permission — silently skip
            self._pinned[channel.id] = msg
        except Exception as e:
            print(f"[FaceManager] Embed update failed: {e}")
 
        # Avatar swap — only on idle↔active boundary, rate-limited to ~2/hr
        if agent_state.crossed_idle_boundary():
            now = time.time()
            if now - self._last_avatar_swap > 1800:
                await self._swap_avatar(new_state)
                self._last_avatar_swap = now
 
    async def _swap_avatar(self, state: str):
        img_path = get_face_image(state)
        if not img_path:
            return
        try:
            with open(img_path, "rb") as f:
                await bot.user.edit(avatar=f.read())
            print(f"[FaceManager] Avatar swapped to '{state}'")
        except Exception as e:
            print(f"[FaceManager] Avatar swap failed: {e}")
 
face_mgr = FaceManager()
 
# ──────────────────────────────────────────────────────────────────────────────
# Chat memory (per Discord channel, separate file from Pi's chat_memory.json)
# ──────────────────────────────────────────────────────────────────────────────
 
def load_memory(channel_id: int) -> list:
    try:
        with open(MEMORY_FILE) as f:
            return json.load(f).get(str(channel_id), [])
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
# Web search
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
    messages   = [
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
# Piper TTS  (blocking — run in executor)
# ──────────────────────────────────────────────────────────────────────────────
 
def tts_to_wav(text: str) -> Path | None:
    voice_model = cfg.get("voice_model", "")
    piper_exe   = PIPER_DIR / "piper"
    if not piper_exe.exists() or not Path(voice_model).exists():
        print("[TTS] Piper or voice model not found")
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out = Path(tmp.name)
    try:
        result = subprocess.run(
            [str(piper_exe), "--model", voice_model, "--output_file", str(out)],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[TTS] Piper error: {result.stderr.decode()}")
            out.unlink(missing_ok=True)
            return None
        return out
    except Exception as e:
        print(f"[TTS] Exception: {e}")
        out.unlink(missing_ok=True)
        return None
 
# ──────────────────────────────────────────────────────────────────────────────
# Shared pipeline: think → (optionally speak) → return reply text
# ──────────────────────────────────────────────────────────────────────────────
 
async def run_pipeline(
    user_input:   str,
    channel_id:   int,
    face_channel: discord.TextChannel | None,
    vc:           discord.VoiceClient | None = None,
) -> str:
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
        wav_path = await asyncio.get_event_loop().run_in_executor(
            None, tts_to_wav, reply
        )
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
# Bot + intents
# ──────────────────────────────────────────────────────────────────────────────
 
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states    = True
 
bot = commands.Bot(command_prefix="!", intents=intents)
 
# Per-guild tracking
face_channels:   dict[int, discord.TextChannel] = {}
listener_tasks:  dict[int, asyncio.Task]        = {}
 
# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────
 
@bot.event
async def on_ready():
    load_wakeword_model()
    print(f"[Bot] Online as {bot.user}")
    print(f"[Bot] Model  : {cfg['text_model']}")
    print(f"[Bot] Wake   : {'loaded ✅' if oww_model else 'not found ❌'}")
 
 
@bot.command(name="ask", help="Ask the bot a question via text.")
async def cmd_ask(ctx: commands.Context, *, question: str):
    face_channel = face_channels.get(ctx.guild.id, ctx.channel)
    await face_mgr.update(face_channel, "listening")
    async with ctx.typing():
        reply = await run_pipeline(
            user_input   = question,
            channel_id   = ctx.channel.id,
            face_channel = face_channel,
        )
    for chunk in [reply[i:i+2000] for i in range(0, len(reply), 2000)]:
        await ctx.reply(chunk)
 
 
@bot.command(name="facechannel", help="Set this channel as the face display channel.")
async def cmd_facechannel(ctx: commands.Context):
    face_channels[ctx.guild.id] = ctx.channel
    await ctx.send(f"Face channel set to {ctx.channel.mention}.")
    await face_mgr.update(ctx.channel, "idle")
 
 
@bot.command(name="join", help="Join your voice channel and start wake word detection.")
async def cmd_join(ctx: commands.Context):
    if not ctx.author.voice:
        await ctx.send("You're not in a voice channel!")
        return
 
    # Disconnect from any existing VC first
    if ctx.voice_client:
        if ctx.guild.id in listener_tasks:
            listener_tasks[ctx.guild.id].cancel()
        await ctx.voice_client.disconnect()
 
    vc           = await ctx.author.voice.channel.connect()
    face_channel = face_channels.get(ctx.guild.id)
 
    if oww_model is None:
        await ctx.send(
            f"Joined **{vc.channel.name}**, but `wakeword.onnx` wasn't found.\n"
            "Wake word detection is disabled — use `!ask` for text chat instead."
        )
        return
 
    fc_note = f"Face updates in {face_channel.mention}." if face_channel \
              else "Tip: use `!facechannel` in a text channel to enable the face embed."
 
    await ctx.send(
        f"Joined **{vc.channel.name}**. 👂 Listening for the wake word continuously.\n{fc_note}"
    )
 
    # Cancel any old listener and start a fresh one
    if ctx.guild.id in listener_tasks:
        listener_tasks[ctx.guild.id].cancel()
 
    task = asyncio.create_task(
        voice_listener_loop(vc, face_channel, ctx.guild.id)
    )
    listener_tasks[ctx.guild.id] = task
 
 
@bot.command(name="leave", help="Leave the voice channel and stop listening.")
async def cmd_leave(ctx: commands.Context):
    if ctx.guild.id in listener_tasks:
        listener_tasks[ctx.guild.id].cancel()
        del listener_tasks[ctx.guild.id]
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel.")
    else:
        await ctx.send("I'm not in a voice channel.")
 
 
@bot.command(name="clearmemory", help="Wipe conversation memory for this channel.")
async def cmd_clearmemory(ctx: commands.Context):
    save_memory(ctx.channel.id, [])
    await ctx.reply("Memory cleared for this channel. 🧹")
 
 
@bot.command(name="status", help="Show current bot state.")
async def cmd_status(ctx: commands.Context):
    face_channel = face_channels.get(ctx.guild.id)
    in_vc        = ctx.voice_client is not None
    ww_active    = ctx.guild.id in listener_tasks
    await ctx.send(
        f"**State:** {agent_state.current}\n"
        f"**Model:** {cfg['text_model']}\n"
        f"**Wake word:** {'loaded ✅' if oww_model else 'not found ❌'}\n"
        f"**Voice:** {'connected' if in_vc else 'not connected'}"
        f"{' — wake word loop active 👂' if ww_active else ''}\n"
        f"**Face channel:** {face_channel.mention if face_channel else 'not set — use `!facechannel`'}"
    )
 
 
# ──────────────────────────────────────────────────────────────────────────────
# @mention → !ask
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
