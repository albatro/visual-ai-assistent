"""
3D avatar demo.

Usage:
    python demo_3d.py                          # opens browser, idle animation
    python demo_3d.py audio.wav                # plays audio with lip sync
    python demo_3d.py audio.wav --loop         # loops audio
    python demo_3d.py audio.wav --no-browser   # server only (open manually)
"""
import sys
import time
import argparse
import threading
import webbrowser
from pathlib import Path

from avatar_server import start_servers_background, play_audio_file, stop_audio

WEB_DIR = str(Path(__file__).parent / 'web')
AVATAR_URL = 'http://localhost:8766/index.html'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', nargs='?', help='Audio file to play')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--no-browser', action='store_true',
                        dest='no_browser')
    args = parser.parse_args()

    print('[demo] Starting servers…')
    start_servers_background(WEB_DIR)
    time.sleep(0.8)   # give servers a moment to bind

    if not args.no_browser:
        print(f'[demo] Opening {AVATAR_URL}')
        webbrowser.open(AVATAR_URL)

    print('[demo] Server running.')
    print('       Open browser at:', AVATAR_URL)
    print('       Press Ctrl+C to quit.\n')

    if args.audio:
        audio = Path(args.audio)
        if not audio.exists():
            print(f'[demo] Audio file not found: {audio}')
            sys.exit(1)

        print(f'[demo] Playing {audio}  (loop={args.loop})')
        try:
            while True:
                play_audio_file(str(audio))
                if not args.loop:
                    break
                time.sleep(0.3)
        except KeyboardInterrupt:
            stop_audio()
    else:
        print('[demo] No audio — avatar in idle mode.')
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    print('[demo] Bye.')


if __name__ == '__main__':
    main()
