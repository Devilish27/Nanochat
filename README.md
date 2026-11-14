# Nanochat

A tiny, fast, portable chat client for [OpenRouter](https://openrouter.ai) and
[Google Gemini](https://aistudio.google.com) (Google AI Studio keys) models on Windows.
One small exe, no installer, no account, no telemetry. It is not an Electron app, so it
stays light: the exe is about 6 MB.

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
- **Saved chats.** Every chat is saved automatically. Reopen, rename (double-click) or delete
  them from the sidebar.
- **Any OpenRouter model.** The full model list loads automatically, with a filter box and a
  free-model label.
- **Google AI Studio (Gemini) keys.** Paste a Gemini API key and the Gemini models your key can
  use appear in the same picker, in their own group. Use either key or both, and even switch
  between providers inside one chat. Favorites, the filter box and saved chats work for both.
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

1. Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys), at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (Gemini), or both.
2. In the app, open **Settings** (bottom left) and paste the key into its field. It is saved
   for next time. Gemini models appear in the picker as soon as the Gemini key is entered.
3. Pick a model, type a message, press Enter (Shift+Enter for a new line).

To quit, use **Settings, Quit app**. It also closes itself about two and a half minutes after
you close the browser tab. Starting it again while it runs just reopens the page.

## Privacy and security

- Your browser talks **directly to openrouter.ai** and, only if you add a Gemini key, directly to
  **generativelanguage.googleapis.com** (Google). The local program never sees your prompts or
  replies; it only stores your saved chats and settings on disk. What Google and OpenRouter do
  with your prompts is governed by their own terms.
- Each key is sent only to its own provider, and the Gemini key goes in a request header, never in the URL.
- The local server listens on `127.0.0.1` only, and rejects requests from other websites.
- Your **API keys are stored as plain text** in `data/settings.json`. Do not share or commit
  that folder. If you build inside a clone of this repo, make sure `data/` is in your `.gitignore`.
- There is no analytics, update check or other network traffic besides OpenRouter and (if you use it) Google Gemini.

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

The page pings `/api/ping` every 10 seconds, and the program exits when pings stop for 150
seconds. Files are written atomically (temp file, then rename), so a crash will not corrupt them.

Endpoints: `GET/PUT /api/settings`, `GET /api/chats`, `GET/PUT/DELETE /api/chats/<id>`,
`GET /api/ping`, `POST /api/quit`.

### Data files

- `data/settings.json`: `{ "key", "gkey", "favorites": [model ids], "model", "effort", "theme" }`
  (`key` is the OpenRouter key, `gkey` the Gemini key, `theme` one of the names above in lower
  case, for example `nord`, or `auto`). Gemini models are stored with a
  `gemini:` prefix, for example `gemini:gemini-2.5-flash`. Settings from version 1.0 load as they are.
- `data/chats/<id>.json`: `{ "id", "title", "model", "updated", "messages": [ { "role", "content", "reasoning"?, "secs"? } ] }`

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

## Notes

Reasoning tokens are billed as output tokens, so higher effort costs more and is slower.
Gemini chats send only the visible text of earlier turns back to the model (not Google's
encrypted thought signatures), which is all plain text chat needs.
This project is not affiliated with OpenRouter or Google.

## Changelog

**1.1**
- Replies are now rendered as Markdown instead of showing raw `#`, `*` and `|` characters.
  Rendering is done by a small built-in renderer (no libraries, still fully offline).
- Fix: code blocks no longer have a stray gap around the code.

**1.0** First release.



## License

MIT (see `LICENSE`).
