#!/usr/bin/env python3
"""Drive a NATIVE interactive bash under a real pty and print its master-side transcript.

The oracle half of the interactive differential (#802 rung 3): the same keystrokes the temen harness
feeds into the #797 terminal are typed here, with the same protocol — **wait for the prompt, then
type the next chunk** — so the two transcripts (prompt on fd 2, the terminal's echo, command output
on fd 1, all interleaved in arrival order) are comparable byte-for-byte.

    pty_oracle.py <bash-binary> <chunk>...

Each `<chunk>` is a Python-escaped byte string (`echo hi\\n`, `\\x03`, `\\x04`); a chunk is written
to the pty only once the transcript so far ends with the prompt (`PS1`, fixed below to the harness's
value). After the last chunk the master is drained to EOF. The transcript is written to stdout as
raw bytes.

Terminal setup mirrors the temen line discipline so the byte streams line up: canonical mode with
echo (the kernel's defaults), but `ONLCR` off (no `\\r\\n` translation of output and echoed
newlines) and `ECHOCTL` off (a `^C` is not echoed as `^C`) — both are output cosmetics the #797
discipline never had. Environment: `PATH`/`HOME`/`PS1`/`TERM` exactly as the harness sets them;
`HISTFILE` empty so the session writes no history file.
"""
import os
import pty
import select
import sys
import termios
import time

PS1 = b"$ "
ENV = {
    "PATH": "/bin:/usr/bin",
    "HOME": "/",
    "PS1": PS1.decode(),
    "TERM": "dumb",
    "HISTFILE": "",
}


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
        os.execve(bash, [bash, "--norc", "--noprofile", "-i"], ENV)
        os._exit(127)

    transcript = bytearray()
    deadline = time.monotonic() + 20.0
    # Bytes already in the transcript when the last chunk was typed: the next prompt must arrive
    # AFTER them (the transcript still ends with the previous prompt while the echo is in flight).
    typed_at = -1

    def pump(until_prompt: bool) -> bool:
        """Read master output into the transcript. `until_prompt`: stop once bytes received since
        the last chunk was typed end with PS1. Returns False on EOF."""
        while True:
            if until_prompt and len(transcript) > typed_at and transcript.endswith(PS1):
                return True
            if time.monotonic() > deadline:
                sys.stderr.write("pty_oracle: timeout\n")
                return False
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

    alive = True
    for chunk in chunks:
        if not pump(until_prompt=True):
            alive = False
            break
        typed_at = len(transcript)
        os.write(master, chunk)
    if alive:
        pump(until_prompt=False)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    sys.stdout.buffer.write(bytes(transcript))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
