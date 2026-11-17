# Nanochat

A tiny, fast, portable chat client for [OpenRouter](https://openrouter.ai) and
[Google Gemini](https://aistudio.google.com) (Google AI Studio keys) models on Windows.
One small exe, no installer, no account, no telemetry. It is not an Electron app, so it
stays light: the exe is about 6 MB.

Since version 1.2 it can also run small language models **on your own computer**: no key, no
account and no internet once the model is downloaded. See [Local models](#local-models).

<img width="1130" height="900" alt="screenshot" src="https://github.com/user-attachments/assets/2e55e2cc-735b-41c4-8715-20ce31ba9cdb" />

## Download

1. Go to the [**Releases**](https://github.com/Devilish27/OpenRouter-Chat/releases) page and download the latest zip.
2. Unzip it anywhere (a USB stick works too).
3. Double-click `Nanochat.exe`. Your browser opens the chat.

Windows SmartScreen may warn about an unsigned app the first time. Click **More info**, then
**Run anyway**. If you would rather not trust a binary, [build it yourself](#build-from-source)
in one command.

## Features

- **Fully portable.** Settings, favorites and chats live in a `data` folder next to the exe.
  Copy the folder to move everything to another PC.
- **Saved chats.** Every chat is saved automatically. Reopen, rename (pencil button or
  double-click, then Enter) or delete them from the sidebar. Delete asks once, in place.
- **Any OpenRouter model.** The full model list loads automatically, with a filter box and a
  free-model label.
- **Google AI Studio (Gemini) keys.** Paste a Gemini API key and the Gemini models your key can
  use appear in the same picker, in their own group. Use either key or both, and even switch
  between providers inside one chat. Favorites, the filter box and saved chats work for both.
- **Local models, built in.** Nanochat contains its own CPU engine for `.safetensors` models
  (TinyLM, Llama, SmolLM, Qwen2/3, Mistral, Gemma and GPT-2; INT8, BF16, FP16 or FP32 weights).
  Replies stream token by token, with the speed and context use shown under each reply.
- **Model library.** Under **Local models** you can import a model from the HuggingFace Hub or
  from a `.zip` URL, upload one from your disk, read its model card, load or unload it from RAM
  and remove it.
- **Local servers.** Enter the address of an OpenAI-compatible server (Ollama, LM Studio,
  llama.cpp) in **Settings** and its models appear in the picker too.
- **Sampling controls.** For local models: prompt format, temperature, top-p, top-k, repetition
  penalty, reply length and a system prompt (the **Params** button).
- **Works on small screens.** On a narrow window the sidebar becomes a slide-in panel.
- **Favorite models.** Click the star next to the model picker. Favorites are listed first.
- **Formatted replies.** Markdown is rendered as the reply streams in: headings, bold, italics,
  strikethrough, nested lists, task lists, tables (they scroll sideways when wide), blockquotes,
  rules, links and code blocks with a Copy button. Your own messages are always shown exactly as
  you typed them. Math (LaTeX) and footnotes are not rendered.
- **Safe by design.** Replies are untrusted text. Raw HTML or scripts in a reply are shown as text
  and never run, links open in a new tab and only `http`, `https` and `mailto` links are
  clickable, and images are never loaded (a remote image would let a reply track you), so they
  appear as plain links instead.
- **Themes.** Pick a look in **Settings**: Auto (follows your system light/dark setting), Light,
  Paper, Sky, Rose, Dark, Midnight, Nord, Forest, Violet, Terminal or High contrast. The choice
  is saved and applied from the very first paint, so there is no flash at startup.
- **Thinking and effort control.** For models that support reasoning, pick Default, Off or an
  effort level (minimal up to xhigh, whatever the model allows). Thinking streams into a
  collapsible "Thought for N s" section and is saved with the chat. For Gemini this shows
  Google's thought summaries, and the levels offered follow what each Gemini model accepts.
- **Streaming replies** with a Stop button.
- **Smart scrolling.** The view follows new text only while you are at the bottom. Scroll up
  and it stays put, with a "Latest" button to jump back.
- **Code blocks with Copy buttons**, and a **Download .html** button when the AI writes a
  complete HTML document.
- Follows your system light or dark mode.

## Quick start

No key at hand? Open **Local models**, pick the **Import URL / Hub** tab, choose the
SmolLM2-135M-Instruct quick pick (about 260 MB) and press **Import from Hub**. When it is
installed, type a message. Everything below is only needed for cloud models.

1. Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys), at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (Gemini), or both.
2. In the app, open **Settings** (bottom left) and paste the key into its field. It is saved
   for next time. Gemini models appear in the picker as soon as the Gemini key is entered.
3. Pick a model, type a message, press Enter (Shift+Enter for a new line).

To quit, use **Settings, Quit app**. It also closes itself about two and a half minutes after
you close the browser tab. Starting it again while it runs just reopens the page.

## Local models

Nanochat runs these models itself, on the CPU, in plain Go. There is nothing else to install.

**Getting a model.** A model is a folder with three files: `model.safetensors` (the weights, one
file), `config.json` and `tokenizer.json` (the HuggingFace "fast" tokenizer). `tokenizer_config.json`
and a `README.md` or `MODEL_CARD.md` are used when present. There are four ways to add one:

- **HuggingFace Hub**: enter a repository such as `HuggingFaceTB/SmolLM2-135M-Instruct`.
- **Zip URL**: a link to a `.zip` holding the files, at the top level or inside one folder.
- **Upload**: drop the zip, or the loose files, on the Upload tab.
- **By hand**: copy the folder into `data/models/` next to the exe. It appears the next time the
  list is opened. This is also how you move models to another PC.

**What runs.** TinyLM, Llama-family models (Llama, SmolLM, Mistral, Qwen2, Qwen3, Gemma) and GPT-2,
stored as INT8 (with per-row scales), BF16, FP16 or FP32. Models over 110 million parameters are
converted to 8-bit weights while loading, which needs a quarter of the memory. GGUF files, models
split over several weight files, and vision or audio models are not supported.

**What to expect.** This is a CPU engine without hand-written vector code, so think small: a
135M-parameter model gives roughly 15 to 60 tokens per second depending on the processor, and a
model twice the size is half as fast. Imports are limited to 900 MB, the context to 2048 tokens,
and loaded models to about 1.2 GB of RAM in total (the least recently used one is unloaded first).
For bigger models use a local server such as Ollama.

**Prompt format.** *Auto* picks the chat template from the model's own special tokens (ChatML,
Llama 3, Zephyr, Mistral), and falls back to a plain "User: / Assistant:" dialogue, or
"Question: / Answer:" for TinyLM. You can force Q&A, Chat, ChatML or Raw completion. Text a
model writes between `<think>` and `</think>` is shown in the Thinking section.

**Local servers.** The page talks to the server directly, so the server must allow requests from
`http://127.0.0.1:47821`. Ollama does by default; in LM Studio switch on CORS in the server settings.
Temperature, top-p, reply length and the system prompt from **Params** are sent to it as well.

## Privacy and security

- Your browser talks **directly to openrouter.ai** and, only if you add a Gemini key, directly to
  **generativelanguage.googleapis.com** (Google). The local program never sees your prompts or
  replies; it only stores your saved chats and settings on disk. What Google and OpenRouter do
  with your prompts is governed by their own terms.
- Each key is sent only to its own provider, and the Gemini key goes in a request header, never in the URL.
- The local server listens on `127.0.0.1` only, and rejects requests from other websites.
- **Local models never leave your computer.** Prompts and replies of a built-in model stay inside the
  program. A local server (Ollama, LM Studio) is contacted only at the address you entered.
- **Importing a model downloads from the address you give** (or from `huggingface.co` and the
  servers it redirects to). Only public `http(s)` addresses are accepted, every redirect is checked
  again, and local or private network addresses are refused. Archives are checked before anything is
  installed: unsafe paths and links are rejected, only model files are kept, and the size limit
  applies to the unpacked data. Model weights in `.safetensors` format are plain numbers and cannot
  run code. A model card is shown with the same safe Markdown rules as a reply.
- Your **API keys are stored as plain text** in `data/settings.json`. Do not share or commit
  that folder. If you build inside a clone of this repo, make sure `data/` is in your `.gitignore`.
- There is no analytics, update check or other network traffic besides OpenRouter, (if you use it) Google Gemini,
  and the model downloads and local server you set up yourself.

## Build from source

You need only the [Go toolchain](https://go.dev/dl/) 1.19 or newer (tested with 1.22).
There are no third-party modules, no Node.js and no C compiler, and no internet is needed to build.

```
winget install GoLang.Go
go build -trimpath -ldflags="-s -w -H=windowsgui" -o Nanochat.exe .
```

- `-H=windowsgui` hides the console window. Leave it out while debugging.
- `-s -w` shrinks the binary, and `-trimpath` removes local paths from it.
- Quick test without building: `go run .` (set `NOOPEN=1` to stop it opening a browser tab).

Cross-compile from Linux or macOS:

```
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -ldflags="-s -w -H=windowsgui" -o Nanochat.exe .
```

The code also builds for Linux and macOS (it opens the default browser with `xdg-open` or
`open`), but only Windows has been tested end to end.

`index.html` is embedded into the exe at compile time, so rebuild after editing it.

## How it works

| File | Role |
| --- | --- |
| `main.go` | Local server on `127.0.0.1` (ports 47821 to 47830). Serves the page and saves settings and chats as JSON. |
| `index.html` | The whole UI: HTML, CSS and JavaScript, with no libraries and no build step. |
| `models.go` | The model library in `data/models` (import, upload, delete) and the `/api/local/` endpoints. |
| `engine.go` | The inference engine: model loading, the transformer forward pass with a key/value cache, sampling. |
| `tokenizer.go` | HuggingFace `tokenizer.json` support (byte-level BPE, SentencePiece BPE, Unigram) and streaming decoding. |
| `safetensors.go` | Reads `.safetensors` files and converts weights (BF16, FP16, INT8) on load. |
| `prompt.go` | Prompt formats, stop sequences and the `<think>` splitter. |

The page pings `/api/ping` every 10 seconds, and the program exits when pings stop for 150
seconds. Files are written atomically (temp file, then rename), so a crash will not corrupt them.

Endpoints: `GET/PUT /api/settings`, `GET /api/chats`, `GET/PUT/DELETE /api/chats/<id>`,
`GET /api/ping`, `POST /api/quit`.

Local models: `GET /api/local/models`, `GET/DELETE /api/local/models/<id>`,
`POST /api/local/models/<id>/load` (`{"action": "load" | "unload"}`), `POST /api/local/models/import`
(`{"source": "huggingface" | "url", "repo" | "url", "name"?, "stream"?}`, progress as server-sent events),
`POST /api/local/models/upload` (multipart form, field `files`), and `POST /api/local/chat`, which
accepts an OpenAI-style chat completion request (`model`, `messages`, `stream`, `temperature`, `top_p`,
`top_k`, `repetition_penalty`, `max_tokens`, `prompt_format`, `system_prompt`) and streams
OpenAI-style chunks. The last chunk carries `nanochat_stats` (tokens, speed, context use).

A reply being generated or a download in progress counts as activity, so the program does not exit
while it is working even if the browser slows the page down in a background tab.

### Data files

- `data/settings.json`: `{ "key", "gkey", "localServerUrl", "localServerKey", "favorites": [model ids], "model", "effort", "theme", "localParams" }`
  (`key` is the OpenRouter key, `gkey` the Gemini key, `theme` one of the names above in lower
  case, for example `nord`, or `auto`). Gemini models are stored with a
  `gemini:` prefix, for example `gemini:gemini-2.5-flash`, built-in local models with `local:` and
  models of a local server with `server:`. `localParams` is `{ "temperature", "topP", "topK",
  "repeatPenalty", "maxTokens", "promptFormat", "systemPrompt" }`. Settings from versions 1.0 and 1.1 load as they are.
- `data/chats/<id>.json`: `{ "id", "title", "model", "updated", "messages": [ { "role", "content", "reasoning"?, "secs"?, "model"?, "stats"? } ] }`
  (`model` is the model that wrote the reply; `stats` is kept for local replies)
- `data/models/<id>/`: one folder per local model with its files, plus `nanochat-model.json`,
  where Nanochat remembers the display name, the source and what it found when it inspected the model.
  Folders starting with `_tmp_` are unfinished imports and are removed at the next start.

## Troubleshooting

- **SmartScreen or antivirus flags the exe.** It is unsigned. Building it yourself from this
  source is the cleanest answer.
- **Nothing opens.** Browse to <http://127.0.0.1:47821> manually.
- **The page can't reach the server.** The program exited (it quits shortly after the last tab
  closes). Start the exe again.
- **"Could not load Gemini models: ..."** The key was rejected or the request failed. Check the key
  at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- **Gemini replies with a quota or 429 error.** Free keys have rate limits. Wait a moment, pick a
  lighter model, or enable billing in Google AI Studio.
- **A Gemini model is missing from the picker.** Only models that support chat (`generateContent`)
  are listed; embedding, speech and image-generation models are hidden on purpose.
- **A thinking level is rejected.** Google changes which levels each Gemini model accepts. Switch the
  setting back to Default, which sends no level and is the safest choice.
- **No thinking shown for a model.** Some models (for example OpenAI's o-series) reason
  internally but do not return their reasoning. The effort setting still affects the answer.
- **Build error mentioning a Go version or toolchain.** Update Go to 1.19 or newer.
- **"Unsupported model architecture".** The built-in engine runs the model families listed under
  [Local models](#local-models). Use a local server such as Ollama for anything else.
- **"Archive is missing tokenizer.json".** The model only ships a slow (SentencePiece or vocab file)
  tokenizer. Look for a repository of the same model that includes `tokenizer.json`.
- **A local model is slow.** Speed depends on model size and CPU. Try a smaller model, close other
  heavy programs, and keep **Max tok** modest. The first reply also includes loading the model.
- **A local model answers nonsense.** Tiny models do that. Check that **Format** is Auto, lower the
  temperature, or try a model trained for chat (an "Instruct" or "Chat" variant).
- **"Could not reach local server".** Check the address (Ollama: `http://127.0.0.1:11434`) and that
  the server allows requests from `http://127.0.0.1:47821` (CORS).
- **"Downloads from private or local network hosts are blocked".** Imports only fetch from the public
  internet. Download the file yourself and use the Upload tab, or copy the folder into `data/models`.

## Notes

Reasoning tokens are billed as output tokens, so higher effort costs more and is slower.
Gemini chats send only the visible text of earlier turns back to the model (not Google's
encrypted thought signatures), which is all plain text chat needs.
This project is not affiliated with OpenRouter or Google.

## Changelog

**1.2**
- Local models: a built-in CPU engine for `.safetensors` models (TinyLM, Llama, SmolLM, Qwen2/3,
  Mistral, Gemma, GPT-2), still with no third-party code.
- Model library: import from the HuggingFace Hub or a zip URL, upload, model cards, load and unload.
- Local servers: models of an OpenAI-compatible server (Ollama, LM Studio) in the picker.
- Sampling parameters and prompt formats for local models; speed and context use under each reply.
- Chats are renamed and deleted in place; the sidebar slides in on small screens.

**1.1**
- Replies are now rendered as Markdown instead of showing raw `#`, `*` and `|` characters.
  Rendering is done by a small built-in renderer (no libraries, still fully offline).
- Fix: code blocks no longer have a stray gap around the code.

**1.0** First release.



## License

MIT (see `LICENSE`).
