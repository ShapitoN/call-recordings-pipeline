#!/usr/bin/env python3
"""Локальная расшифровка русскоязычных созвонов и краткий разбор через Claude CLI.

  calls.py run                 обработать очередь
  calls.py transcribe A B      расшифровать файл A в текст B
  calls.py status              показать состояние очереди
"""
import os
import urllib.request
import urllib.parse
from contextlib import contextmanager
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RECORDS = Path(os.environ.get("CALLS_RECORDS", str(ROOT / "recordings"))).expanduser()
INBOX = RECORDS / "Входящие"
MODELS = Path(os.environ.get("CALLS_MODELS", str(ROOT / "models"))).expanduser()
EXTERNAL_SUMMARIES = Path(os.environ.get("CALLS_EXTERNAL_SUMMARIES", str(ROOT / "external-summaries"))).expanduser()

MODEL, EFFORT = "sonnet", "medium"
THREADS = 6
RATE = 16000
MIN_WORDS = 40          # меньше — в записи нет разговора, разбирать нечего
RETRY_MINUTES = 25      # пауза перед второй попыткой разбора, дальше вдвое длиннее каждый раз
MAX_TRIES = 6           # ~26 ч попыток; лимит подписки сбрасывается за 5 ч, а сломанный ответ не жжёт токены сутки
CALENDAR_ENABLED = False   # Автоматическое создание событий отключено.
CAL_MODEL = "haiku"
CAL = "mcp__claude_ai_Google_Calendar__"
CAL_TOOLS = [CAL + "list_events", CAL + "create_event"]
# Остальные коннекторы claude.ai не нужны, а их описания — ~120 тыс. токенов на запрос.
CAL_OFF = ["mcp__claude_ai_Notion", "mcp__claude_ai_Figma", "mcp__claude_ai_Claude_Docs",
           "mcp__claude_ai_Gmail", "mcp__claude_ai_Google_Drive", "mcp__plugin_telegram_telegram",
           *(CAL + t for t in ("list_calendars", "get_event", "search_events", "update_event",
                               "delete_event", "respond_to_event", "suggest_time"))]
WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

STATE = ".конвейер.json"
MEDIA_EXT = {".webm", ".mp4", ".mkv", ".mov", ".m4a", ".mp3", ".ogg", ".opus", ".oga", ".wav",
             ".aac", ".flac", ".3gp", ".amr", ".mka"}
PARTIAL_EXT = {".crdownload", ".part", ".partial", ".download", ".tmp"}
DAY_FORMAT = "%d.%m.%Y"
UNTITLED = re.compile(r"^\d{2}-\d{2}(-\d{2})?(-\d+)?$")
ME, THEM = "Я", "Собеседник"


def log(*args):
    print(*args, flush=True)


# ---------- состояние папки встречи ----------

def load_state(meeting):
    try:
        return json.loads((meeting / STATE).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_state(meeting, state):
    (meeting / STATE).write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def now():
    return datetime.now().isoformat(timespec="seconds")


def notify(text):
    """Telegram опционален. Без настроек уведомление выводится локально."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log(text)
        return None
    payload = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    # Не выводим URL с токеном даже при ошибке сети.
    try:
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=payload)
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.load(response)
        if not result.get("ok"):
            raise RuntimeError("Telegram rejected notification")
        return result["result"]["message_id"]
    except Exception:
        raise RuntimeError("Telegram notification failed; check credentials and connectivity") from None


# ---------- звук ----------

def ffprobe(media, *args):
    out = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json", str(media)],
                         check=True, capture_output=True, text=True, encoding="utf-8").stdout
    return json.loads(out)


def duration(media):
    try:
        return float(ffprobe(media, "-show_entries", "format=duration")["format"]["duration"])
    except (KeyError, ValueError, subprocess.CalledProcessError):
        return 0.0


def audio_tracks(media):
    """Номера дорожек микрофона и звука ноутбука, если запись сделана с раздельными голосами."""
    streams = ffprobe(media, "-select_streams", "a", "-show_entries", "stream=index:stream_tags=title")["streams"]
    titles = [(s.get("tags") or {}).get("title", "").lower() for s in streams]
    mic = next((i for i, t in enumerate(titles) if "микрофон" in t), None)
    sys_ = next((i for i, t in enumerate(titles) if "звук" in t), None)
    return (mic, sys_) if mic is not None and sys_ is not None else None


def decode(media, track=None):
    cmd = ["ffmpeg", "-v", "error", "-i", str(media)]
    if track is not None:
        cmd += ["-map", f"0:a:{track}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ---------- расшифровка ----------

@dataclass
class Seg:
    start: float
    end: float
    text: str
    who: str = ""


class Recognizer:
    def __init__(self):
        import sherpa_onnx
        m = MODELS / "giga-am-v3"
        self.sherpa = sherpa_onnx
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(m / "encoder.int8.onnx"), decoder=str(m / "decoder.onnx"),
            joiner=str(m / "joiner.onnx"), tokens=str(m / "tokens.txt"),
            num_threads=THREADS, model_type="nemo_transducer")

    def segments(self, audio, who=""):
        cfg = self.sherpa.VadModelConfig()
        cfg.silero_vad.model = str(MODELS / "silero_vad.onnx")
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.5
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = 20   # GigaAM обучен на фразах до ~25 с
        cfg.sample_rate = RATE
        vad = self.sherpa.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
        chunks = []

        def drain():
            while not vad.empty():
                chunks.append((vad.front.start, np.array(vad.front.samples, dtype=np.float32)))
                vad.pop()

        win = cfg.silero_vad.window_size
        for i in range(0, len(audio), win):
            vad.accept_waveform(audio[i:i + win])
            drain()
        vad.flush()
        drain()
        out = []
        for start, samples in chunks:
            stream = self.rec.create_stream()
            stream.accept_waveform(RATE, samples)
            self.rec.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                out.append(Seg(start / RATE, (start + len(samples)) / RATE, text, who))
        return out


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def drop_echo(mine, theirs):
    """Без наушников микрофон слышит собеседника из колонок: такие куски — эхо, а не речь пользователя."""
    kept = []
    for seg in mine:
        heard = set()
        for other in theirs:
            if other.start - 1.5 < seg.end and seg.start < other.end + 1.5:
                heard |= words(other.text)
        own = words(seg.text)
        # Короткое «да», «угу» совпадает с чем угодно — его не трогаем, пусть лучше останется дубль.
        if len(own) >= 3 and len(own & heard) / len(own) >= 0.5:
            continue
        kept.append(seg)
    return kept


def merge_turns(segs):
    """Подряд идущие фразы одного человека — в одну реплику."""
    out = []
    for seg in segs:
        last = out[-1] if out else None
        if last and last.who == seg.who and seg.start - last.end < 1.5 and seg.end - last.start < 60:
            last.text += " " + seg.text
            last.end = seg.end
        else:
            out.append(Seg(seg.start, seg.end, seg.text, seg.who))
    return out


def stamp(sec):
    sec = int(sec)
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def transcribe(media, out):
    """Пишет расшифровку и возвращает True, если голоса размечены."""
    t0 = time.time()
    rec = Recognizer()
    tracks = audio_tracks(media)
    if tracks:
        mine = rec.segments(decode(media, tracks[0]), ME)
        theirs = rec.segments(decode(media, tracks[1]), THEM)
        segs = merge_turns(sorted(drop_echo(mine, theirs) + theirs, key=lambda s: s.start))
        lines = [f"{stamp(s.start)} {s.who}: {s.text}" for s in segs]
    else:
        segs = rec.segments(decode(media))
        lines = [f"{stamp(s.start)} {s.text}" for s in segs]
    tmp = out.with_name(out.name + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(out)
    log(f"расшифровано {media}: {len(lines)} реплик за {time.time() - t0:.0f} с")
    return bool(tracks)


# ---------- разбор ----------

def parse_day(name):
    try:
        return datetime.strptime(name, DAY_FORMAT)
    except ValueError:
        return None


def meeting_start(meeting):
    m = re.match(r"(\d{2})-(\d{2})", meeting.name)
    day = parse_day(meeting.parent.name)
    return day.replace(hour=int(m[1]), minute=int(m[2])) if m else day


def analyze(meeting, media, transcript, state):
    start = meeting_start(meeting)
    minutes = round(duration(media) / 60)
    voices = (f"размечены: «{ME}» — микрофон пользователя, «{THEM}» — звук с ноутбука "
              f"(если собеседников несколько, все они «{THEM}»)" if state.get("voices")
              else "не размечены — кто пользователь, определи по смыслу")
    head = [f"Встреча: {start:%d.%m.%Y} ({WEEKDAYS[start.weekday()]}), начало {start:%H:%M}, "
            f"длительность {minutes} мин.",
            f"Голоса: {voices}."]
    if state.get("source"):
        head.append(f"Исходное имя файла: {state['source']}")
    text = "\n".join(head) + "\n\nРасшифровка:\n" + transcript
    cmd = [shutil.which("claude") or "claude", "-p", "--model", MODEL, "--effort", EFFORT,
           "--system-prompt", (ROOT / "prompt.md").read_text(encoding="utf-8"),
           "--tools", "", "--setting-sources", "", "--strict-mcp-config",
           "--no-session-persistence", "--output-format", "json"]
    r = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", timeout=1200, cwd=ROOT / "logs")
    try:
        reply = json.loads(r.stdout)
    except ValueError:
        raise RuntimeError((r.stderr or r.stdout or f"код {r.returncode}").strip()[-300:])
    if r.returncode or reply.get("is_error"):
        raise RuntimeError(str(reply.get("result") or reply)[-300:])
    body = reply["result"]
    try:
        data = json.loads(body[body.find("{"):body.rfind("}") + 1])
    except ValueError:
        raw = ROOT / "logs" / f"{meeting.parent.name}_{meeting.name}.ответ.txt"
        raw.write_text(body, encoding="utf-8")
        raise RuntimeError(f"ответ не разобрался как JSON, сохранён в {raw}")
    tokens = log_usage(reply, f"{meeting.parent.name}/{meeting.name}", minutes)
    return data, start, minutes, tokens


def log_usage(reply, what, minutes):
    usage = reply.get("usage") or {}
    tokens_in = sum(usage.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    with open(ROOT / "logs" / "usage.tsv", "a", encoding="utf-8") as f:
        f.write(f"{now()}\t{what}\t{minutes}\t{tokens_in}\t"
                f"{usage.get('output_tokens', 0)}\t{reply.get('total_cost_usd', '')}\n")
    return tokens_in + usage.get("output_tokens", 0)


# ---------- календарь ----------

CAL_PROMPT = """Ты добавляешь встречи в основной Google Calendar пользователя. Для каждой встречи из списка:
1. list_events за день встречи: startTime — день T00:00:00, endTime — следующий день T00:00:00
   (и timeZone, если он есть у встречи).
2. Если в этот день уже есть событие про ту же встречу (то же время или пересекается, и совпадает
   компания, человек или тема; для встречи без времени — та же компания или человек в этот день),
   ничего не создавай: status "exists", id и note — название найденного события.
3. Иначе create_event ровно с полями из event, ничего не меняя и не добавляя: без участников,
   без Google Meet. status "created", id созданного события.
Ошибка инструмента — status "error", note — причина. Ничего не удаляй и не меняй.
Верни только JSON-массив без пояснений: [{"n": 1, "status": "created", "id": "…", "note": ""}]"""


def planned_events(data, call_start, call_title):
    """Будущие встречи из разбора → поля create_event. Прошедшие отбрасываются, без даты — только в Telegram."""
    out = []
    for m in data.get("meetings") or []:
        title = (m.get("title") or "").strip() or "Встреча"
        try:
            day = datetime.strptime((m.get("date") or "").strip(), "%d.%m.%Y")
        except ValueError:
            out.append({"event": {"summary": title}, "when": "дата не названа", "status": "nodate"})
            continue
        time_m = re.fullmatch(r"(\d{1,2})[:.](\d{2})", (m.get("time") or "").strip())
        if time_m:
            begin = day.replace(hour=int(time_m[1]), minute=int(time_m[2]))
            minutes = int(m.get("minutes") or 60) if str(m.get("minutes") or "").isdigit() else 60
            end = begin + timedelta(minutes=max(15, min(minutes, 480)))
            if begin <= datetime.now() and not m.get("tz"):
                continue
            event = {"summary": title, "startTime": f"{begin:%Y-%m-%dT%H:%M:%S}",
                     "endTime": f"{end:%Y-%m-%dT%H:%M:%S}"}
            when_text = f"{WEEKDAYS[begin.weekday()]} {begin:%d.%m %H:%M}"
        else:
            if day.date() < datetime.now().date():
                continue
            event = {"summary": f"{title} (уточнить {'время' if m.get('exact') is not False else 'дату и время'})", "allDay": True,
                     "startTime": f"{day:%Y-%m-%dT00:00:00}",
                     "endTime": f"{day + timedelta(days=1):%Y-%m-%dT00:00:00}"}
            when_text = f"{WEEKDAYS[day.weekday()]} {day:%d.%m}, время не назвали"
        if m.get("exact") is False:
            if time_m:
                event["summary"] += " (дата примерная)"
            when_text += ", дата примерная"
        if m.get("tz"):
            event["timeZone"] = m["tz"]
            when_text += f" ({m['tz']})"
        at = f", минута {m['minute']}" if m.get("minute") else ""
        event["description"] = "<br>".join(x for x in [
            f"Договорились на созвоне {call_start:%d.%m.%Y %H:%M} «{call_title}»{at}.",
            f"Кто: {m['with']}" if m.get("with") else "",
            f"О чём: {m['about']}" if m.get("about") else "",
            "Ссылку на встречу добавить.",
        ] if x)
        out.append({"event": event, "when": when_text})
    return out


def add_to_calendar(meeting, items):
    """Создаёт события через коннектор Google Calendar. Возвращает список результатов по номерам."""
    for item in items:
        if item.get("status"):
            continue
        item["event"]["description"] = re.sub(r"<br>Разбор: .*", "", item["event"]["description"])
        item["event"]["description"] += f"<br>Разбор: {meeting / 'разбор.md'}"
    todo = [{"n": i + 1, "event": item["event"]} for i, item in enumerate(items) if not item.get("status")]
    if not todo:
        return
    cmd = [shutil.which("claude") or "claude", "-p", "--model", CAL_MODEL, "--system-prompt", CAL_PROMPT,
           "--tools", "", "--setting-sources", "", "--no-session-persistence", "--output-format", "json",
           "--permission-mode", "dontAsk", "--allowedTools", *CAL_TOOLS, "--disallowedTools", *CAL_OFF]
    r = subprocess.run(cmd, input=json.dumps(todo, ensure_ascii=False, indent=1),
                       capture_output=True, text=True, encoding="utf-8", timeout=600, cwd=ROOT / "logs")
    try:
        reply = json.loads(r.stdout)
    except ValueError:
        raise RuntimeError((r.stderr or r.stdout or f"код {r.returncode}").strip()[-300:])
    if r.returncode or reply.get("is_error"):
        raise RuntimeError(str(reply.get("result") or reply)[-300:])
    log_usage(reply, f"{meeting.parent.name}/{meeting.name} · календарь", 0)
    body = reply["result"]
    try:
        results = json.loads(body[body.find("["):body.rfind("]") + 1])
    except ValueError:
        raise RuntimeError(f"ответ календаря не разобрался: {body[-200:]}")
    for res in results:
        n = res.get("n")
        if isinstance(n, int) and 1 <= n <= len(items) and res.get("status") in ("created", "exists"):
            items[n - 1].update(status=res["status"], id=res.get("id", ""), note=res.get("note", ""))
    missing = [i for i in items if not i.get("status")]
    if missing:
        errors = "; ".join(r.get("note", "") for r in results if r.get("status") == "error")
        raise RuntimeError(errors or f"не создано: {len(missing)}")


def calendar_step(meeting, state):
    """Догоняет календарь по встрече. Возвращает True, если пора сказать мне об окончательной неудаче."""
    items = state.get("calendar") or []
    if not items or all(i.get("status") for i in items):
        return False
    tries = state.get("calendar_tries", 0)
    if tries >= MAX_TRIES:
        return False
    last = state.get("calendar_last_try")
    if last and datetime.now() - datetime.fromisoformat(last) < timedelta(minutes=RETRY_MINUTES * 2 ** (tries - 1)):
        return False
    state["calendar_last_try"] = now()
    state["calendar_tries"] = tries + 1
    try:
        add_to_calendar(meeting, items)
        state.pop("calendar_error", None)
        log(f"календарь {meeting}: " + ", ".join(f"{i['status']} {i['event']['summary']}" for i in items))
    except Exception as error:
        state["calendar_error"] = str(error)[-300:]
        log(f"календарь не удался {meeting} (попытка {tries + 1}): {error}")
        return tries + 1 >= MAX_TRIES
    return False


def calendar_lines(state):
    lines = []
    for i in state.get("calendar") or []:
        name = i["event"]["summary"]
        if i.get("status") == "created":
            lines.append(f"• {i['when']} — {name}")
        elif i.get("status") == "nodate":
            lines.append(f"• {name} — дату не назвали, в календарь не добавил")
        elif i.get("status") == "exists":
            lines.append(f"• {i['when']} — уже было в календаре: {i.get('note') or name}")
        else:
            lines.append(f"• {i['when']} — {name}: не добавилось, повторю позже")
    return lines


def add_tasks(meeting, data, start, state):
    """Сохраняет задачи рядом с разбором; повторный экспорт заменяет тот же файл."""
    tasks = []
    for task in data.get("my_tasks", []):
        if task.get("task"):
            tasks.append(dict(task, kind="my"))
    for task in data.get("their_tasks", []):
        if task.get("dev") and task.get("task"):
            tasks.append({"task": task.get("check") or f"Проверить: {task['task']}",
                          "due": task.get("due", ""), "kind": "check"})
    (meeting / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state["tasks_added"] = len(tasks)


def bullets(items, fmt):
    return "\n".join(fmt(i) for i in items if i) or "—"


def task_line(t):
    due = f" — {t['due']}" if t.get("due") else ""
    at = f" ({t['minute']})" if t.get("minute") else ""
    return f"- [ ] {t.get('task', '')}{due}{at}"


def render(data, start, minutes, tokens, external_summary_path):
    parts = [
        f"# {data.get('title') or 'Созвон'}",
        f"{start:%d.%m.%Y}, {start:%H:%M}, {minutes} мин · {data.get('kind', '')}",
        f"Участники: {data.get('participants') or '—'}",
        "## Коротко", data.get("summary") or "—",
        "## Что сделать мне", bullets(data.get("my_tasks", []), task_line),
        "## Что обещали другие",
        bullets(data.get("their_tasks", []),
                lambda t: f"- {t.get('who', '')}: {t.get('task', '')}" + (f" — {t['due']}" if t.get("due") else "")),
        "## Договорённости", bullets(data.get("agreements", []), lambda a: f"- {a}"),
        "## Важные слова собеседника",
        bullets(data.get("quotes", []), lambda q: f"- [{q.get('minute', '')}] «{q.get('text', '')}» — {q.get('why', '')}"),
        "## Открытые вопросы", bullets(data.get("open_questions", []), lambda q: f"- {q}"),
        "## Следующий контакт", data.get("next_contact") or "—",
        "## Встречи на будущее",
        bullets(data.get("meetings", []),
                lambda m: f"- {m.get('date', '')} {m.get('time', '')} — {m.get('title', '')}"
                          + (f" ({m['minute']})" if m.get("minute") else "")),
    ]
    if external_summary_path:
        parts.append(f"Итоги внешнего помощника за эту встречу: `{external_summary_path}`")
    parts.append(f"---\nРасшифровка: `расшифровка.txt` · разбор: Claude {MODEL}, {tokens / 1000:.0f} тыс. токенов")
    return "\n\n".join(parts) + "\n"


def telegram_text(data, start, minutes, folder, calendar=()):
    lines = [f"Созвон разобран: {data.get('title') or 'без названия'}",
             f"{start:%d.%m}, {start:%H:%M}, {minutes} мин", "", data.get("summary") or ""]
    tasks = [t for t in data.get("my_tasks", []) if t.get("task")]
    if tasks:
        lines += ["", "Что сделать тебе:"]
        lines += [f"• {t['task']}" + (f" — {t['due']}" if t.get("due") else "") for t in tasks]
    if data.get("next_contact"):
        lines += ["", f"Дальше: {data['next_contact']}"]
    if calendar:
        lines += ["", "Календарь (ссылку на встречу добавь сам):", *calendar]
    lines += ["", f"Папка: {folder}"]
    return "\n".join(lines)


def external_summary(start, minutes):
    """Итоги внешнего помощника за то же время: он уже прислал «что сделать» в Telegram, второй раз не шлём."""
    for path in EXTERNAL_SUMMARIES.glob("*— итоги.md"):
        try:
            at = datetime.strptime(path.name[:19], "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        if start - timedelta(minutes=20) <= at <= start + timedelta(minutes=minutes + 1):
            return path
    return None


def safe_title(title):
    title = re.sub(r'[\\/:*?"<>|\n\r\t]+', " ", title or "").strip(" .")
    return re.sub(r"\s+", " ", title)[:60].rstrip()


# ---------- обход папок ----------

def find_media(meeting):
    return next((p for p in sorted(meeting.iterdir())
                 if p.is_file() and p.stem == "запись" and p.suffix.lower() in MEDIA_EXT), None)


def meetings():
    if not RECORDS.is_dir():
        return []
    days = sorted((d for d in RECORDS.iterdir() if d.is_dir() and parse_day(d.name)), key=lambda d: parse_day(d.name))
    return [m for day in days for m in sorted(day.iterdir()) if m.is_dir() and find_media(m)]


def settled(path, wait=5):
    """Файл дописан: размер не меняется несколько секунд."""
    if time.time() - path.stat().st_mtime > 60:
        return True
    size = path.stat().st_size
    time.sleep(wait)
    return path.stat().st_size == size


def when(path):
    m = re.search(r"(20\d\d)[-_.]?(\d\d)[-_.]?(\d\d)[ _T-]+(\d\d)[-_.:h]?(\d\d)", path.name)
    if m:
        try:
            return datetime(*map(int, m.groups()))
        except ValueError:
            pass
    try:
        created = ffprobe(path, "-show_entries", "format_tags=creation_time")["format"]["tags"]["creation_time"]
        return datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
    except (KeyError, ValueError, subprocess.CalledProcessError):
        return datetime.fromtimestamp(path.stat().st_mtime)


def place(path, **state):
    """Кладёт файл в «Screen Record/ДД.ММ.ГГГГ/ЧЧ-ММ/запись.*» и возвращает папку встречи."""
    at = when(path)
    day = RECORDS / at.strftime(DAY_FORMAT)
    meeting = day / f"{at:%H-%M}"
    n = 2
    while meeting.exists():
        meeting = day / f"{at:%H-%M}-{n}"
        n += 1
    meeting.mkdir(parents=True)
    shutil.move(str(path), meeting / f"запись{path.suffix.lower()}")
    save_state(meeting, {"source": path.name, "imported": now(), **state})
    log(f"{path.name} → {meeting}")
    return meeting


def take_inbox():
    INBOX.mkdir(parents=True, exist_ok=True)
    for path in sorted(INBOX.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() in PARTIAL_EXT:
            continue
        if path.suffix.lower() not in MEDIA_EXT:
            continue
        if settled(path):
            place(path)


def process(meeting):
    """Один шаг по встрече. Возвращает True, если что-то сделано."""
    state = load_state(meeting)
    media = find_media(meeting)
    if not settled(media):
        return False
    transcript_path = meeting / "расшифровка.txt"
    if not transcript_path.exists():
        if state.get("transcribe_error"):
            return False
        try:
            state["voices"] = transcribe(media, transcript_path)
            state["transcribed"] = now()
        except Exception as error:
            state["transcribe_error"] = str(error)[-300:]
            log(f"не расшифровалось {meeting}: {error}")
            notify(f"Запись не расшифровалась: {error}\n\nПапка: {meeting}")
        save_state(meeting, state)
        return True

    if (meeting / "разбор.md").exists():
        if CALENDAR_ENABLED and state.get("calendar") and not all(i.get("status") for i in state["calendar"]):
            gave_up = calendar_step(meeting, state)
            save_state(meeting, state)
            if gave_up:
                notify("Не смог добавить в календарь встречу с созвона — добавь сам:\n"
                       + "\n".join(calendar_lines(state)).replace(": не добавилось, повторю позже", "")
                       + f"\n\nОшибка: {state.get('calendar_error', '')}\nПапка: {meeting}")
            elif all(i.get("status") for i in state["calendar"]) and state.pop("calendar_late", None):
                notify("Добавил в календарь встречу с созвона (ссылку добавь сам):\n"
                       + "\n".join(calendar_lines(state)) + f"\n\nПапка: {meeting}")
            save_state(meeting, state)
        return False
    if state.get("analysis_skip"):
        return False
    transcript = transcript_path.read_text(encoding="utf-8")
    if len(re.sub(r"\[\d+:\d\d\]", "", transcript).split()) < MIN_WORDS:
        state["analysis_skip"] = "в записи почти нет речи"
        save_state(meeting, state)
        log(f"пропуск разбора {meeting}: почти нет речи")
        return True
    tries = state.get("analysis_tries", 0)
    if tries >= MAX_TRIES:
        return False
    last = state.get("analysis_last_try")
    if last and datetime.now() - datetime.fromisoformat(last) < timedelta(minutes=RETRY_MINUTES * 2 ** (tries - 1)):
        return False

    state["analysis_last_try"] = now()
    state["analysis_tries"] = tries + 1
    try:
        data, start, minutes, tokens = analyze(meeting, media, transcript, state)
    except Exception as error:
        state["analysis_error"] = str(error)[-300:]
        log(f"разбор не удался {meeting} (попытка {tries + 1}): {error}")
        if tries + 1 >= MAX_TRIES:
            notify(f"Созвон расшифрован, но разбор так и не получился: {state['analysis_error']}\n\n"
                   f"Расшифровка в папке: {meeting}")
        save_state(meeting, state)
        return False

    external_summary_path = external_summary(start, minutes)
    (meeting / "разбор.md").write_text(render(data, start, minutes, tokens, external_summary_path), encoding="utf-8")
    state.update(analyzed=now(), title=data.get("title", ""), tokens=tokens)
    state.pop("analysis_error", None)
    title = safe_title(data.get("title"))
    if title and UNTITLED.match(meeting.name):
        target = meeting.with_name(f"{meeting.name} {title}")
        if not target.exists():
            meeting = meeting.rename(target)
    if not state.get("silent"):
        try:
            add_tasks(meeting, data, start, state)
        except Exception as error:   # задачи — не повод терять разбор и сообщение
            log(f"задачи не добавились {meeting}: {error}")
        if CALENDAR_ENABLED:
            state["calendar"] = planned_events(data, start, data.get("title") or "созвон")
            calendar_step(meeting, state)
    calendar = calendar_lines(state)
    if any(not i.get("status") for i in state.get("calendar") or []):
        state["calendar_late"] = True   # когда догонит — одно короткое сообщение
    if state.get("silent"):
        log(f"разобрано без уведомления {meeting}")
    elif external_summary_path:
        log(f"разобрано {meeting}; итоги уже прислал внешний помощник ({external_summary_path.name})")
        if calendar:
            notify(f"Созвон {start:%d.%m %H:%M}, календарь (ссылку на встречу добавь сам):\n"
                   + "\n".join(calendar))
    elif not state.get("notified"):
        state["telegram_id"] = notify(telegram_text(data, start, minutes, meeting, calendar))
        state["notified"] = now()
    save_state(meeting, state)
    log(f"разобрано {meeting}: {tokens} токенов")
    return True


@contextmanager
def exclusive_lock(path):
    """Блокировка очереди на Windows и POSIX."""
    with open(path, "a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run():
    (ROOT / "logs").mkdir(exist_ok=True)
    with exclusive_lock(ROOT / "logs" / ".lock"):
        while True:
            take_inbox()
            # По одному шагу за круг: новая запись, пришедшая во время работы, не ждёт всю очередь.
            if not any(process(m) for m in meetings()):
                break


def status():
    for m in meetings():
        s = load_state(m)
        stage = ("разобрано" if (m / "разбор.md").exists() else
                 s.get("analysis_skip") or
                 (f"разбор не удался: {s['analysis_error']}" if s.get("analysis_error") else
                  "ждёт разбора" if (m / "расшифровка.txt").exists() else
                  f"ошибка расшифровки: {s['transcribe_error']}" if s.get("transcribe_error") else
                  "ждёт расшифровки"))
        print(f"{m.parent.name}/{m.name}: {stage}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "transcribe" and len(sys.argv) == 4:
        transcribe(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "status":
        status()
    else:
        print(__doc__)
        sys.exit(2)
