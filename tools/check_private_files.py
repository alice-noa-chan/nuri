"""Local Git privacy checks; this tool does not install hooks or change Git state.

Pre-commit checks the index, so forced additions and staged blob contents count.
Pre-push reads Git's ref-update lines from stdin and checks outgoing history.
Optional UTF-8 literal markers live in .private/provenance-markers.txt, one per
line (blank lines and # comments are ignored). Matching is case-insensitive.
Violations never print paths, content, marker values, or captured Git errors.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROTECTED_ROOTS = {"data", ".private", ".dev", ".watch"}
OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.IGNORECASE)


class CheckError(Exception):
    """A check could not be completed; details stay out of hook output."""


class PrivacyGuard:
    def __init__(self, repo: str | Path, markers: str | Path | None = None):
        self.repo = Path(repo).resolve()
        self.repo = Path(self.git("rev-parse", "--show-toplevel").decode().strip())
        marker_file = Path(markers) if markers is not None else Path(".private/provenance-markers.txt")
        if not marker_file.is_absolute():
            marker_file = self.repo / marker_file
        try:
            marker_file.resolve().relative_to((self.repo / ".private").resolve())
            text = marker_file.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            if markers is not None:
                raise CheckError from None
            text = ""
        except (OSError, UnicodeError, ValueError):
            raise CheckError from None
        self.markers = tuple(line.strip().casefold() for line in text.splitlines()
                             if line.strip() and not line.lstrip().startswith("#"))
        self.protected_paths = 0
        self.marked_contents = 0
        self.checked_blobs: set[bytes] = set()
        self.checked_commits: set[str] = set()

    def git(self, *args: str, optional: bool = False) -> bytes | None:
        try:
            result = subprocess.run(["git", "-C", str(self.repo), *args],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except OSError:
            raise CheckError from None
        if result.returncode:
            if optional:
                return None
            raise CheckError
        return result.stdout

    @staticmethod
    def protected(path: bytes) -> bool:
        root = path.split(b"/", 1)[0].decode("utf-8", errors="surrogateescape").casefold()
        return root in PROTECTED_ROOTS

    def check_content(self, content: bytes, message: bool = False) -> None:
        if not self.markers:
            return
        if not message and b"\0" in content:
            return
        try:
            text = content.decode("utf-8-sig", errors="replace" if message else "strict").casefold()
        except UnicodeError:
            return  # Binary blobs are outside the optional public-text check.
        if any(marker in text for marker in self.markers):
            self.marked_contents += 1

    def check_blob(self, oid: bytes) -> None:
        if self.markers and oid not in self.checked_blobs:
            self.checked_blobs.add(oid)
            self.check_content(self.git("cat-file", "blob", oid.decode("ascii")))

    def check_index(self) -> None:
        changed = set(self.git("diff", "--cached", "--name-only", "-z",
                               "--diff-filter=ACMRT").split(b"\0"))
        for entry in self.git("ls-files", "--stage", "-z").split(b"\0"):
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            mode, oid, stage = metadata.split()
            if stage != b"0":
                raise CheckError
            if self.protected(path):
                self.protected_paths += 1
            elif path in changed and mode != b"160000":
                self.check_blob(oid)

    def check_commit(self, oid: str) -> None:
        if oid in self.checked_commits:
            return
        self.checked_commits.add(oid)
        for entry in self.git("ls-tree", "-r", "-z", "--full-tree", oid).split(b"\0"):
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            _, kind, blob = metadata.split()
            if self.protected(path):
                self.protected_paths += 1
            elif kind == b"blob":
                self.check_blob(blob)
        if self.markers:
            self.check_content(self.git("show", "-s", "--format=%B", oid), message=True)

    def check_push(self, updates: str) -> None:
        """Check every outgoing commit, including files removed in a later commit.

        New refs or unavailable remote tips use the complete local ancestry.
        Existing remote objects bound the check to commits not at that tip.
        No remote is fetched or contacted by this check.
        """
        self.check_index()
        for line in updates.splitlines():
            fields = line.split()
            if len(fields) != 4:
                raise CheckError
            _, local_oid, _, remote_oid = fields
            if not OBJECT_ID.fullmatch(local_oid) or not OBJECT_ID.fullmatch(remote_oid):
                raise CheckError
            if set(local_oid) == {"0"}:
                continue  # A ref deletion sends no local objects.
            kind = self.git("cat-file", "-t", local_oid).strip()
            # Annotated tag messages can disclose provenance independently of
            # the commit they refer to. Follow a chain of annotated tags too.
            target = local_oid
            while kind == b"tag":
                tag = self.git("cat-file", "-p", target)
                self.check_content(tag, message=True)
                target = tag.splitlines()[0].removeprefix(b"object ").decode("ascii")
                if not OBJECT_ID.fullmatch(target):
                    raise CheckError
                kind = self.git("cat-file", "-t", target).strip()
            if kind != b"commit":
                raise CheckError  # Non-commit ref targets need a separate audit.
            args = ["rev-list", target]
            if set(remote_oid) != {"0"} and self.git("cat-file", "-e", f"{remote_oid}^{{commit}}",
                                                    optional=True) is not None:
                args.append(f"^{remote_oid}")
            for commit in self.git(*args).decode("ascii").splitlines():
                self.check_commit(commit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository or directory inside it")
    parser.add_argument("--markers", help="optional marker file inside the repository's .private/")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-commit", action="store_true", help="check the index (default)")
    mode.add_argument("--pre-push", action="store_true", help="read Git ref updates from stdin")
    mode.add_argument("--commit-msg", metavar="PATH", help="check a local commit-message file")
    mode.add_argument("--audit-head", action="store_true", help="check the current commit's tree and message")
    args = parser.parse_args(argv)
    try:
        guard = PrivacyGuard(args.repo, args.markers)
        if args.pre_push:
            guard.check_push(sys.stdin.read())
        elif args.commit_msg:
            guard.check_content(Path(args.commit_msg).read_bytes(), message=True)
        elif args.audit_head:
            guard.check_commit(guard.git("rev-parse", "--verify", "HEAD").decode("ascii").strip())
        else:
            guard.check_index()
    except (CheckError, OSError, UnicodeError, ValueError):
        print("Privacy check could not be completed; inspect the local repository and private configuration.",
              file=sys.stderr)
        return 2
    if guard.protected_paths or guard.marked_contents:
        print(f"Privacy check failed: {guard.protected_paths} protected path(s), "
              f"{guard.marked_contents} content marker match(es).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
