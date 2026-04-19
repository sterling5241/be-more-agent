# Discord Bot Setup

## 1. Install new dependencies

```bash
source venv/bin/activate
pip install "discord.py[voice]" PyNaCl aiohttp --break-system-packages
```

Also make sure `ffmpeg` is installed (needed for voice playback):
```bash
sudo apt install ffmpeg -y
```

---

## 2. Create your Discord bot

1. Go to https://discord.com/developers/applications
2. Click **New Application** → give it a name
3. Go to **Bot** tab → click **Add Bot**
4. Under **Privileged Gateway Intents**, enable:
   - ✅ Message Content Intent
   - ✅ Server Members Intent
5. Copy your **Token** (you'll need it in step 4)
6. Go to **OAuth2 → URL Generator**:
   - Scopes: `bot`
   - Bot permissions:
     - Send Messages
     - Embed Links
     - Attach Files
     - Read Message History
     - Manage Messages  ← needed for pinning face embed
     - Connect
     - Speak
     - Use Voice Activity
7. Copy the generated URL and open it to invite the bot to your server

---

## 3. Make sure Ollama is running

```bash
ollama serve
# In another terminal, confirm your model is pulled:
ollama pull gemma3:1b
```

---

## 4. Run the bot

```bash
export DISCORD_TOKEN=your_token_here
source venv/bin/activate
python discord_bot.py
```

---

## 5. Usage in Discord

### Setup (do once)
Go to the channel where you want the face to appear and type:
```
!facechannel
```
This pins a face embed in that channel and updates it with every state change.

### Text chat
```
!ask what's the weather like today?
@YourBot what time is it in Tokyo?
```

### Voice
```
!join          — bot joins your current voice channel
!listen        — records 5 seconds, transcribes with Whisper, replies in VC with Piper TTS
!leave         — bot leaves voice channel
```

### Utility
```
!clearmemory   — wipe conversation history for this channel
!status        — show current state, model, face channel
```

---

## Face images

The bot reads from the same `faces/` folder as `agent.py`:
```
faces/
  idle/        ← shown when bot is waiting
  listening/   ← shown while recording / reading your message
  thinking/    ← shown while Ollama is processing
  speaking/    ← shown while TTS is playing / reply is sent
  error/        ← shown on errors
```
Put any `.png` files in each subfolder. The bot picks one at random per state.

**Avatar** swaps on the first idle↔active transition, then rate-limits to ~once per 30 minutes (Discord enforces ~2 avatar changes/hour globally).

---

## Tips

- Run `agent.py` (Pi mode) and `discord_bot.py` side by side if you want — they use separate memory files (`chat_memory.json` vs `discord_memory.json`).
- Memory is per-channel — each Discord channel has its own conversation history.
- If Whisper or Piper aren't set up yet, text commands (`!ask`, @mention) still work fine — voice features just won't be available.
