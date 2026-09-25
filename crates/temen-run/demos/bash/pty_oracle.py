#!/usr/bin/env python3
"""Drive a NATIVE interactive bash under a real pty and print its master-side transcript.

The oracle half of the interactive differential (#802 rung 3): the same keystrokes the temen harness
feeds into the #797 terminal are typed here, with the same protocol — **wait for the prompt, then
type the next chunk** — so the two transcripts (prompt on fd 2, the terminal's echo, command output
on fd 1, all interleaved in arrival order) are comparable byte-for-byte.

    pty_oracle.py <bash-binary> <chunk>...

Each `<chunk>` is a Python-escaped byte string (`echo hi\\n`, `\\x03`, `\\x04`); a chunk is written
to the pty only once bash is **idle at a prompt** (see `idle_at_prompt`). After the last chunk the
master is drained to EOF. The transcript is written to stdout as raw bytes. A session that stops
making progress kills bash and exits 3 — it never waits on it unbounded.

Terminal setup mirrors the temen line discipline so the byte streams line up: canonical mode with
echo (the kernel's defaults), but `ONLCR` off (no `\\r\\n` translation of output and echoed
newlines) and `ECHOCTL` off (a `^C` is not echoed as `^C`) — both are output cosmetics the #797
discipline never had — and the personality's 80×24 winsize. Environment: `PATH`/`HOME`/`PS1`
exactly as the harness sets them; `HISTFILE` empty so the session writes no history file;
`TERM`/`TERMCAP` inherited from the caller (#1496 — the harness exports the personality's fixed
entry, `temen_posix::TERMCAP_ENTRY`, on both sides; `TERM=dumb` when unset).
"""
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import time

PS1 = b"$ "
ENV = {
    "PATH": "/bin:/usr/bin",
    "HOME": "/",
    "PS1": PS1.decode(),
    "TERM": os.environ.get("TERM", "dumb"),
    "HISTFILE": "",
}
if "TERMCAP" in os.environ:
    ENV["TERMCAP"] = os.environ["TERMCAP"]


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: pty_oracle.py <bash-binary> <chunk>...\n")
        return 2
    bash = sys.argv[1]
    chunks = [
        c.encode("utf-8").decode("unicode_escape").encode("latin-1") for c in sys.argv[2:]
    ]
    pid, master = pty.fork()
    if pid == 0:
        # Child: the slave is fds 0/1/2. Turn off the two output cosmetics before exec.
        attrs = termios.tcgetattr(0)
        attrs[1] &= ~termios.ONLCR  # c_oflag
        attrs[3] &= ~termios.ECHOCTL  # c_lflag
        termios.tcsetattr(0, termios.TCSANOW, attrs)
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        os.execve(bash, [bash, "--norc", "--noprofile", "-i"], ENV)
        os._exit(127)

    transcript = bytearray()
    # Bytes already in the transcript when the last chunk was typed: the next prompt must arrive
    # AFTER them (the transcript still ends with the previous prompt while the echo is in flight).
    typed_at = -1

    def idle_at_prompt() -> bool:
        """Whether bash is waiting for the next keystroke: everything it wrote since the last chunk
        has been read, it ends with PS1, and bash is asleep — the terminal read, since nothing else
        blocks it there.

        The prompt text alone is not enough, and each gap has cost a nightly run. Readline repaints
        the prompt mid-line (Home on a wrapped line redraws `\\e[A\\r$ `), and a key typed then
        can land after the line is accepted, while bash is between lines in cooked mode; the
        kernel stores a cooked `^D` as NUL, which readline, raw again, reads as `C-@`. And readline
        draws the prompt before it blocks in its read: a `^C` in that gap is handled without the
        interrupted read that makes bash redraw the prompt. Either way bash waits at a prompt the
        protocol never sees."""
        if not (len(transcript) > typed_at and transcript.endswith(PS1)):
            return False
        if select.select([master], [], [], 0)[0]:
            return False  # more output pending: bash is still writing (or blocked draining it)
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        return state[:1] in ("S", "I")  # interruptible sleep (`I`: macOS, asleep > 20 s)

    def pump(until_prompt: bool) -> bool:
        """Read master output into the transcript. `until_prompt`: stop once bash is idle at a
        prompt. Returns False on EOF; exits 3 if bash makes no progress for 20 s."""
        deadline = time.monotonic() + 20.0
        while True:
            if until_prompt and idle_at_prompt():
                return True
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                sys.stderr.write(
                    "pty_oracle: no %s within 20 s; transcript so far: %r\n"
                    % ("prompt" if until_prompt else "exit", bytes(transcript))
                )
                sys.exit(3)
            r, _, _ = select.select([master], [], [], 0.05)
            if not r:
                continue
            try:
                data = os.read(master, 4096)
            except OSError:
                return False  # EIO: the slave side closed (bash exited)
            if not data:
                return False
            transcript.extend(data)

    for chunk in chunks:
        if not pump(until_prompt=True):
            sys.stderr.write(
                "pty_oracle: bash exited before the chunk %r; transcript: %r\n"
                % (chunk, bytes(transcript))
            )
            return 3
        typed_at = len(transcript)
        os.write(master, chunk)
    pump(until_prompt=False)
    os.waitpid(pid, 0)  # the slave closed: bash is exiting
    sys.stdout.buffer.write(bytes(transcript))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
