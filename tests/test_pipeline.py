import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import calls


class PipelineTests(unittest.TestCase):
    def test_synthetic_meeting_lifecycle_and_repeat(self):
        """Import -> existing transcript -> mocked analysis -> tasks -> no duplicate analysis."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inbox = root / 'Входящие'
            inbox.mkdir()
            media = inbox / '2030-01-02_10-30_test.wav'
            media.write_bytes(b'synthetic media; not decoded by this test')
            data = {'title': 'Тестовый проект', 'summary': 'Проверка',
                    'my_tasks': [{'task': 'Подготовить документ'}],
                    'their_tasks': [{'who': 'Разработчик', 'task': 'Сделать форму', 'dev': True},
                                    {'who': 'Клиент', 'task': 'Ответить', 'dev': False}]}
            with patch.object(calls, 'RECORDS', root), patch.object(calls, 'INBOX', inbox):
                with patch.object(calls, 'settled', return_value=True):
                    calls.take_inbox()
                    self.assertFalse(media.exists())
                    meeting = calls.meetings()[0]
                    (meeting / 'расшифровка.txt').write_text('[00:00] ' + 'слово ' * 50, encoding='utf-8')
                    with patch.object(calls, 'analyze', return_value=(data, calls.meeting_start(meeting), 1, 100)) as analyze, \
                         patch.object(calls, 'external_summary', return_value=None), \
                         patch.object(calls, 'notify', return_value=123) as notify:
                        self.assertTrue(calls.process(meeting))
                        meeting = calls.meetings()[0]
                        self.assertTrue((meeting / 'разбор.md').is_file())
                        tasks = json.loads((meeting / 'tasks.json').read_text(encoding='utf-8'))
                        self.assertEqual([x['kind'] for x in tasks], ['my', 'check'])
                        self.assertEqual(calls.load_state(meeting)['telegram_id'], 123)
                        self.assertFalse(calls.process(meeting))
                        analyze.assert_called_once()
                        notify.assert_called_once()

    def test_lock_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'
            with self.assertRaises(ValueError):
                with calls.exclusive_lock(path):
                    raise ValueError('synthetic')
            with calls.exclusive_lock(path):
                pass

    def test_windows_lock_acquires_and_releases(self):
        operations = []
        fake = types.SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2,
                                     locking=lambda fd, mode, size: operations.append((mode, size)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lock'
            with patch.dict('sys.modules', {'msvcrt': fake}), patch.object(calls.os, 'name', 'nt'):
                with calls.exclusive_lock(path):
                    pass
            self.assertEqual(operations, [(1, 1), (2, 1)])
            self.assertEqual(path.read_bytes(), b'0')

    def test_telegram_disabled_is_local(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(calls.urllib.request, 'urlopen') as network:
            self.assertIsNone(calls.notify('synthetic notification'))
            network.assert_not_called()

    def test_telegram_failure_does_not_leak_token(self):
        with patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'SECRET_SENTINEL', 'TELEGRAM_CHAT_ID': '1'}):
            with patch.object(calls.urllib.request, 'urlopen', side_effect=OSError('SECRET_SENTINEL')):
                with self.assertRaises(RuntimeError) as caught:
                    calls.notify('test')
                self.assertNotIn('SECRET_SENTINEL', str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)

    def test_short_transcript_is_not_sent_to_claude(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / '02.01.2030' / '10-30'
            folder.mkdir(parents=True)
            (folder / 'запись.wav').write_bytes(b'fake')
            (folder / 'расшифровка.txt').write_text('[00:00] Привет', encoding='utf-8')
            with patch.object(calls, 'settled', return_value=True), patch.object(calls, 'analyze') as analyze:
                self.assertTrue(calls.process(folder))
                analyze.assert_not_called()
                self.assertTrue(calls.load_state(folder)['analysis_skip'])


if __name__ == '__main__':
    unittest.main()
