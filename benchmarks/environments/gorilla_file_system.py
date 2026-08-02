"""A simulated Unix-like file system, matching the BFCL `GorillaFileSystem` environment.

Ported from the reference implementation in
https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/gorilla_file_system.py
(Apache-2.0), since the ground truth shipped with the BFCL dataset was
generated against that exact behavior — any semantic drift here would
silently make a correct model action look wrong. Only ``GorillaFileSystem``,
one of the eleven environment classes BFCL's multi-turn categories use, is
covered (see ``benchmarks/bfcl/fixtures/NOTICE.md`` for why).
"""

from __future__ import annotations

from typing import Any

from benchmarks.errors import InvalidCallError, UnknownFunctionError


class File:
    def __init__(self, name: str, content: str = "") -> None:
        self.name = name
        self.content = content

    def snapshot(self) -> dict[str, Any]:
        return {"type": "file", "content": self.content}


class Directory:
    def __init__(self, name: str, parent: Directory | None = None) -> None:
        self.name = name
        self.parent = parent
        self.contents: dict[str, File | Directory] = {}

    def add_file(self, file_name: str, content: str = "") -> None:
        self.contents[file_name] = File(file_name, content)

    def add_directory(self, dir_name: str) -> Directory:
        new_dir = Directory(dir_name, self)
        self.contents[dir_name] = new_dir
        return new_dir

    def get(self, item_name: str) -> File | Directory | None:
        if item_name == ".":
            return self
        return self.contents.get(item_name)

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "directory",
            "contents": {name: item.snapshot() for name, item in self.contents.items()},
        }


_INVALID_NAME_CHARS = set('|/\\?%*:"><')


def _validate_name(name: str) -> bool:
    return not any(char in _INVALID_NAME_CHARS for char in name)


class GorillaFileSystemEnvironment:
    """The `call`/`snapshot` environment wrapping a simulated file tree."""

    def __init__(self) -> None:
        self.root = Directory("root")
        self._current_dir = self.root

    def load_scenario(self, config: dict[str, Any]) -> None:
        """Seed the file tree from a BFCL `initial_config["GorillaFileSystem"]` block."""
        root_contents = config.get("root", {})
        if not root_contents:
            self._current_dir = self.root
            return
        root_name = next(iter(root_contents))
        self.root = Directory(root_name)
        self._load_directory(root_contents[root_name].get("contents", {}), self.root)
        self._current_dir = self.root

    def _load_directory(self, contents: dict[str, Any], parent: Directory) -> None:
        for name, data in contents.items():
            if data["type"] == "directory":
                child = parent.add_directory(name)
                self._load_directory(data.get("contents", {}), child)
            elif data["type"] == "file":
                parent.add_file(name, data.get("content", ""))

    def snapshot(self) -> Any:
        return self.root.snapshot()

    def call(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        method = getattr(self, name, None)
        if method is None or name.startswith("_") or not callable(method):
            raise UnknownFunctionError(f"GorillaFileSystem has no operation {name!r}")
        try:
            result: dict[str, Any] | None = method(**kwargs)
        except TypeError as exc:
            raise InvalidCallError(f"{name}({kwargs!r}): {exc}") from exc
        return result if result is not None else {}

    # -- operations, matching the reference implementation's signatures and semantics --

    def pwd(self) -> dict[str, Any]:
        path: list[str] = []
        current: Directory | None = self._current_dir
        while current is not None and current is not self.root:
            path.append(current.name)
            current = current.parent
        return {"current_working_directory": "/" + "/".join(reversed(path))}

    def ls(self, a: bool = False) -> dict[str, Any]:
        contents = list(self._current_dir.contents.keys())
        if not a:
            contents = [item for item in contents if not item.startswith(".")]
        return {"current_directory_content": contents}

    def cd(self, folder: str) -> dict[str, Any]:
        folder = folder.rstrip("/") or "/"
        if folder not in {".", "..", "/"} and "/" in folder:
            return {
                "error": (
                    f"cd: {folder}: Unsupported path. Only one folder level at a time is supported."
                )
            }
        if folder == "..":
            if self._current_dir.parent is not None:
                self._current_dir = self._current_dir.parent
                return {}
            if self._current_dir is self.root:
                return {"error": "Current directory is already the root. Cannot go back."}
            return {"error": "cd: ..: No such directory"}

        target = self._navigate(folder)
        if isinstance(target, dict):
            return target
        self._current_dir = target
        return {"current_working_directory": target.name}

    def mkdir(self, dir_name: str) -> dict[str, Any]:
        if not _validate_name(dir_name):
            return {"error": f"mkdir: cannot create directory '{dir_name}': Invalid character"}
        if dir_name in self._current_dir.contents:
            return {"error": f"mkdir: cannot create directory '{dir_name}': File exists"}
        self._current_dir.add_directory(dir_name)
        return {}

    def touch(self, file_name: str) -> dict[str, Any]:
        if not _validate_name(file_name):
            return {"error": f"touch: cannot touch '{file_name}': Invalid character"}
        if file_name in self._current_dir.contents:
            return {"error": f"touch: cannot touch '{file_name}': File exists"}
        self._current_dir.add_file(file_name)
        return {}

    def echo(self, content: str, file_name: str | None = None) -> dict[str, Any]:
        if file_name is None:
            return {"terminal_output": content}
        if not _validate_name(file_name):
            return {"error": f"echo: cannot write to '{file_name}': Invalid character"}
        item = self._current_dir.get(file_name)
        if not isinstance(item, File):
            return {"error": f"echo: cannot write to '{file_name}': No such file"}
        item.content = content
        return {}

    def cat(self, file_name: str) -> dict[str, Any]:
        if not _validate_name(file_name):
            return {"error": f"cat: '{file_name}': Invalid character"}
        item = self._current_dir.get(file_name)
        if item is None:
            return {"error": f"cat: '{file_name}': No such file or directory"}
        if not isinstance(item, File):
            return {"error": f"cat: '{file_name}': Is a directory"}
        return {"file_content": item.content}

    def find(self, path: str = ".", name: str | None = None) -> dict[str, Any]:
        target = self._navigate(path)
        if isinstance(target, dict):
            error = target.get("error", "")
            if error.startswith("cd:"):
                return {"error": error.replace("cd:", "find:", 1)}
            return target

        matches: list[str] = []

        def recurse(directory: Directory, base: str) -> None:
            for item_name, item in directory.contents.items():
                item_path = f"{base}/{item_name}"
                if name is None or name in item_name:
                    matches.append(item_path)
                if isinstance(item, Directory):
                    recurse(item, item_path)

        recurse(target, path.rstrip("/"))
        return {"matches": matches}

    def wc(self, file_name: str, mode: str = "l") -> dict[str, Any]:
        if mode not in {"l", "w", "c"}:
            return {"error": f"wc: invalid mode '{mode}'"}
        item = self._current_dir.get(file_name)
        if not isinstance(item, File):
            return {"error": f"wc: {file_name}: No such file or directory"}
        if mode == "l":
            return {"count": len(item.content.splitlines()), "type": "lines"}
        if mode == "w":
            return {"count": len(item.content.split()), "type": "words"}
        return {"count": len(item.content), "type": "characters"}

    def sort(self, file_name: str) -> dict[str, Any]:
        item = self._current_dir.get(file_name)
        if not isinstance(item, File):
            return {"error": f"sort: {file_name}: No such file or directory"}
        return {"sorted_content": "\n".join(sorted(item.content.splitlines()))}

    def grep(self, file_name: str, pattern: str) -> dict[str, Any]:
        item = self._current_dir.get(file_name)
        if not isinstance(item, File):
            return {"error": f"grep: {file_name}: No such file or directory"}
        matching = [line for line in item.content.splitlines() if pattern in line]
        return {"matching_lines": matching}

    def du(self, human_readable: bool = False) -> dict[str, Any]:
        def size_of(item: File | Directory) -> int:
            if isinstance(item, File):
                return len(item.content.encode("utf-8"))
            return sum(size_of(child) for child in item.contents.values())

        total = size_of(self._current_dir)
        if not human_readable:
            return {"disk_usage": f"{total} bytes"}
        size = float(total)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return {"disk_usage": f"{size:.2f} {unit}"}
            size /= 1024
        return {"disk_usage": f"{size:.2f} PB"}

    def tail(self, file_name: str, lines: int = 10) -> dict[str, Any]:
        item = self._current_dir.get(file_name)
        if not isinstance(item, File):
            return {"error": f"tail: {file_name}: No such file or directory"}
        content_lines = item.content.splitlines()
        lines = min(lines, len(content_lines))
        return {"last_lines": "\n".join(content_lines[-lines:] if lines else [])}

    def diff(self, file_name1: str, file_name2: str) -> dict[str, Any]:
        item1 = self._current_dir.get(file_name1)
        item2 = self._current_dir.get(file_name2)
        if not isinstance(item1, File) or not isinstance(item2, File):
            return {"error": f"diff: {file_name1} or {file_name2}: No such file or directory"}
        lines1 = item1.content.splitlines()
        lines2 = item2.content.splitlines()
        diff_lines = [
            f"- {line1}\n+ {line2}"
            for line1, line2 in zip(lines1, lines2, strict=False)
            if line1 != line2
        ]
        return {"diff_lines": "\n".join(diff_lines)}

    def mv(self, source: str, destination: str) -> dict[str, Any]:
        if source not in self._current_dir.contents:
            return {"error": f"mv: cannot move '{source}': No such file or directory"}
        if "/" in destination:
            return {
                "error": (
                    "mv: path not allowed in destination. Provide only a file or directory name."
                )
            }
        item = self._current_dir.contents[source]
        dest_item = self._current_dir.get(destination)
        if isinstance(dest_item, Directory):
            if source in dest_item.contents:
                return {
                    "error": f"mv: cannot move '{source}' to '{destination}/{source}': File exists"
                }
            del self._current_dir.contents[source]
            self._place(dest_item, source, item)
            return {"result": f"'{source}' moved to '{destination}/{source}'"}
        if dest_item is not None:
            return {"error": f"mv: cannot move '{source}' to '{destination}': Not a directory"}
        del self._current_dir.contents[source]
        self._place(self._current_dir, destination, item)
        return {"result": f"'{source}' moved to '{destination}'"}

    def cp(self, source: str, destination: str) -> dict[str, Any]:
        if source not in self._current_dir.contents:
            return {"error": f"cp: cannot copy '{source}': No such file or directory"}
        if "/" in destination:
            return {
                "error": (
                    "cp: path not allowed in destination. Provide only a file or directory name."
                )
            }
        item = self._current_dir.contents[source]
        dest_item = self._current_dir.get(destination)
        if isinstance(dest_item, Directory):
            if source in dest_item.contents:
                return {
                    "error": f"cp: cannot copy '{source}' to '{destination}/{source}': File exists"
                }
            self._place(dest_item, source, item, copy=True)
            return {"result": f"'{source}' copied to '{destination}/{source}'"}
        if dest_item is not None:
            return {"error": f"cp: cannot copy '{source}' to '{destination}': Not a directory"}
        self._place(self._current_dir, destination, item, copy=True)
        return {"result": f"'{source}' copied to '{destination}'"}

    def rm(self, file_name: str) -> dict[str, Any]:
        if file_name not in self._current_dir.contents:
            return {"error": f"rm: cannot remove '{file_name}': No such file or directory"}
        del self._current_dir.contents[file_name]
        return {"result": f"'{file_name}' removed"}

    def rmdir(self, dir_name: str) -> dict[str, Any]:
        item = self._current_dir.get(dir_name)
        if not isinstance(item, Directory):
            if item is None:
                return {"error": f"rmdir: cannot remove '{dir_name}': No such file or directory"}
            return {"error": f"rmdir: cannot remove '{dir_name}': Not a directory"}
        if item.contents:
            return {"error": f"rmdir: cannot remove '{dir_name}': Directory not empty"}
        del self._current_dir.contents[dir_name]
        return {"result": f"'{dir_name}' removed"}

    def _place(
        self, parent: Directory, name: str, item: File | Directory, *, copy: bool = False
    ) -> None:
        if isinstance(item, File):
            parent.add_file(name, item.content)
        else:
            new_dir = parent.add_directory(name)
            new_dir.contents = {
                child_name: self._clone(child) if copy else child
                for child_name, child in item.contents.items()
            }
            if not copy:
                for child in new_dir.contents.values():
                    if isinstance(child, Directory):
                        child.parent = new_dir

    def _clone(self, item: File | Directory) -> File | Directory:
        if isinstance(item, File):
            return File(item.name, item.content)
        clone = Directory(item.name)
        clone.contents = {name: self._clone(child) for name, child in item.contents.items()}
        for child in clone.contents.values():
            if isinstance(child, Directory):
                child.parent = clone
        return clone

    def _navigate(self, path: str | None) -> Directory | dict[str, Any]:
        if path is None or path == ".":
            return self._current_dir
        if path == "/":
            return self.root

        parts = path.strip("/").split("/")
        current: Directory = self._current_dir if not path.startswith("/") else self.root
        for part in parts:
            candidate = current.get(part)
            if not isinstance(candidate, Directory):
                return {"error": f"cd: '{path}': No such file or directory"}
            current = candidate
        return current


def create(initial_config: dict[str, Any]) -> GorillaFileSystemEnvironment:
    """Build an environment from a BFCL task's ``initial_config``, keyed by class name.

    The ``environment_factory`` callable :class:`~benchmarks.runner.BenchmarkRunner` expects.
    """
    env = GorillaFileSystemEnvironment()
    env.load_scenario(initial_config.get("GorillaFileSystem", {}))
    return env
