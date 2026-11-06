"""Read-only installation checks. Does not contact services or print secrets."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys


def main():
    root = Path(__file__).resolve().parent
    checks = []
    def check(label, passed, detail):
        checks.append(passed)
        print(f"{'OK' if passed else 'MISSING'}: {label}: {detail}")
    check('Python 3.11+', sys.version_info >= (3, 11), sys.version.split()[0])
    for name in ('ffmpeg', 'ffprobe', 'claude'):
        found = shutil.which(name)
        check(name, bool(found), found or 'not found in PATH')
    for module in ('numpy', 'sherpa_onnx'):
        check(module, importlib.util.find_spec(module) is not None, 'Python environment')
    models = Path(os.environ.get('CALLS_MODELS', str(root / 'models'))).expanduser()
    for name in ('silero_vad.onnx', 'giga-am-v3/encoder.int8.onnx', 'giga-am-v3/decoder.onnx',
                 'giga-am-v3/joiner.onnx', 'giga-am-v3/tokens.txt'):
        path = models / name
        check(name, path.is_file() and path.stat().st_size > 0, str(path))
    for label, target in (('Records folder', Path(os.environ.get('CALLS_RECORDS', str(root / 'recordings'))).expanduser()),
                          ('Project/logs folder', root)):
        ancestor = target
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        check(label, ancestor.is_dir() and os.access(ancestor, os.W_OK),
              f'{target} (nearest existing parent writable check only)')
    token, chat = bool(os.environ.get('TELEGRAM_BOT_TOKEN')), bool(os.environ.get('TELEGRAM_CHAT_ID'))
    if token != chat:
        check('Optional Telegram', False, 'both bot token and chat ID are required; values hidden')
    else:
        print('INFO: Telegram configured (not contacted)' if token else 'INFO: Telegram disabled')
    print('INFO: CLI authentication, model loading, Windows execution and real transcription require a separate smoke test.')
    return 0 if all(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
