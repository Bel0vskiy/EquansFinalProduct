import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

RVT_ROOT = Path("rvt_files")
IFC_ROOT = Path("ifc_files")

# Converter args
REGION = "EMEA"
IFCVER = "ifc4"

TAIL_LINES = 10


# ------------------ FS helpers ------------------

def ensure_dirs() -> None:
    if not RVT_ROOT.exists():
        raise FileNotFoundError(f"Directory not found: {RVT_ROOT}.")
    if not RVT_ROOT.is_dir():
        raise NotADirectoryError(f"{RVT_ROOT} exists, but it is not a directory.")
    IFC_ROOT.mkdir(parents=True, exist_ok=True)


def list_subfolders(root: Path) -> List[Path]:
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower())


def list_rvt_files(folder: Path) -> List[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".rvt"),
        key=lambda p: p.name.lower(),
    )


def converter_script_path() -> Path:
    p = Path(__file__).resolve().with_name("rvt_to_ifc.py")
    if not p.is_file():
        raise FileNotFoundError("rvt_to_ifc.py was not found next to convert_menu.py")
    return p


def tail_file(path: Path, n: int) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            lines = data.splitlines()[-n:]
            return [ln.decode("utf-8", errors="replace") for ln in lines]
    except Exception:
        return []


# ------------------ Conversion runner ------------------

def build_cmd(rvt_to_ifc: Path, in_rvt: Path, out_ifc: Path, manifest_path: Path) -> List[str]:
    return [
        sys.executable, "-u", str(rvt_to_ifc),
        str(in_rvt),
        "--region", REGION,
        "--ifcver", IFCVER,
        "--outfile", str(out_ifc),
        "--save-manifest", str(manifest_path),
    ]


def run_convert_one_logged(
    rvt_to_ifc: Path,
    in_rvt: Path,
    out_ifc: Path,
    manifest_path: Path,
    log_path: Path,
    show_progress_cb=None,
) -> Tuple[bool, int]:
    """
    Strictly sequential run (one file at a time).
    Writes stdout+stderr to log_path on disk.
    """
    cmd = build_cmd(rvt_to_ifc, in_rvt, out_ifc, manifest_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        logf.write("COMMAND:\n")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        try:
            while True:
                ret = proc.poll()
                if show_progress_cb is not None:
                    show_progress_cb(tail_file(log_path, TAIL_LINES))
                if ret is not None:
                    return (ret == 0), ret
                time.sleep(0.3)
        except KeyboardInterrupt:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            for _ in range(10):
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise


# ------------------ Simple (no curses) UI ------------------

def pick_index_simple(title: str, items: List[str]) -> Optional[int]:
    if not items:
        return None
    print("\n" + title)
    for i, it in enumerate(items, start=1):
        print(f"{i}) {it}")
    print("0) Back")
    while True:
        raw = input("Choice: ").strip()
        try:
            v = int(raw)
            if v == 0:
                return None
            if 1 <= v <= len(items):
                return v - 1
        except ValueError:
            pass
        print("Invalid input.")


def flow_convert_single_simple() -> None:
    ensure_dirs()
    rvt_to_ifc = converter_script_path()

    folders = list_subfolders(RVT_ROOT)
    if not folders:
        print(f"No subfolders in {RVT_ROOT}.")
        return

    idx = pick_index_simple(f"Pick a folder in {RVT_ROOT}:", [p.name for p in folders])
    if idx is None:
        return

    folder = folders[idx]
    files = list_rvt_files(folder)
    if not files:
        print(f"No .rvt files in {folder}")
        return

    fidx = pick_index_simple(f"Pick a (.rvt) file from {folder.name}:", [p.name for p in files])
    if fidx is None:
        return

    in_rvt = files[fidx]
    out_dir = IFC_ROOT / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_ifc = out_dir / (in_rvt.stem + ".ifc")
    manifest = out_dir / (in_rvt.stem + "_manifest.json")
    log_path = out_dir / (in_rvt.stem + ".log")

    print("\nStarting:")
    print(" ".join(build_cmd(rvt_to_ifc, in_rvt, out_ifc, manifest)))
    print(f"\nLog: {log_path}\n")

    try:
        ok, code = run_convert_one_logged(rvt_to_ifc, in_rvt, out_ifc, manifest, log_path)
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        print(f"Log: {log_path}")
        return

    if ok:
        print("Done")
        print(f"IFC: {out_ifc}")
        print(f"Manifest: {manifest}")
        print(f"Log: {log_path}")
    else:
        print(f"Error (exit code {code})")
        print(f"Log: {log_path}")


def flow_convert_folder_simple() -> None:
    ensure_dirs()
    rvt_to_ifc = converter_script_path()

    folders = list_subfolders(RVT_ROOT)
    if not folders:
        print(f"No subfolders in {RVT_ROOT}.")
        return

    idx = pick_index_simple(
        f"Pick a folder in {RVT_ROOT} (convert all):",
        [p.name for p in folders],
    )
    if idx is None:
        return

    folder = folders[idx]
    files = list_rvt_files(folder)
    if not files:
        print(f"No .rvt files in {folder}")
        return

    out_dir = IFC_ROOT / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFound {len(files)} files. Converting strictly one-by-one.")
    print(f"Output: {out_dir}\n")

    ok_count = 0
    fail_count = 0

    try:
        for i, in_rvt in enumerate(files, start=1):
            out_ifc = out_dir / (in_rvt.stem + ".ifc")
            manifest = out_dir / (in_rvt.stem + "_manifest.json")
            log_path = out_dir / (in_rvt.stem + ".log")

            print(f"[{i}/{len(files)}] {in_rvt.name}")
            print(f"  -> {out_ifc.name}")
            print(f"  log: {log_path.name}")

            ok, code = run_convert_one_logged(rvt_to_ifc, in_rvt, out_ifc, manifest, log_path)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                print(f"  exit code {code} (see {log_path})")

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        print(f"Output: {out_dir}")
        return

    print(f"\nSummary: {ok_count} | {fail_count}")
    print(f"Results: {out_dir}")


def main_simple() -> None:
    while True:
        print("\nRVT -> IFC menu")
        print(f"RVT: {RVT_ROOT}")
        print(f"IFC: {IFC_ROOT}")
        print(f"Args: --region {REGION} --ifcver {IFCVER} --save-manifest (and --outfile)")
        print("1) Convert ONE file")
        print("2) Convert a FOLDER")
        print("3) Exit")

        c = input("Choice: ").strip()
        if c == "1":
            flow_convert_single_simple()
        elif c == "2":
            flow_convert_folder_simple()
        elif c == "3":
            return


# ------------------ Curses UI (arrow keys) ------------------

def has_real_tty() -> bool:
    return bool(os.environ.get("TERM")) and sys.stdin.isatty() and sys.stdout.isatty()


def main() -> None:
    if not has_real_tty():
        main_simple()
        return

    try:
        import curses
    except Exception:
        main_simple()
        return

    def _draw_menu(stdscr, title: str, items: List[str], selected: int, footer: str = "") -> None:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        title_line = f" {title} "
        stdscr.addstr(0, max(0, (w - len(title_line)) // 2), title_line, curses.A_BOLD)
        start_y = 2
        for i, item in enumerate(items):
            y = start_y + i
            if y >= h - 2:
                break
            if i == selected:
                stdscr.addstr(y, 2, f"> {item}", curses.A_REVERSE)
            else:
                stdscr.addstr(y, 2, f"  {item}")
        help_line = footer or "↑/↓ to select, Enter to confirm, Esc to go back"
        stdscr.addstr(h - 1, 2, help_line[: max(0, w - 4)], curses.A_DIM)
        stdscr.refresh()

    def pick_from_list_curses(stdscr, title: str, items: List[str], footer: str = "") -> Optional[int]:
        if not items:
            return None
        selected = 0
        while True:
            _draw_menu(stdscr, title, items, selected, footer=footer)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = (selected - 1) % len(items)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = (selected + 1) % len(items)
            elif key in (10, 13, curses.KEY_ENTER):
                return selected
            elif key in (27,):
                return None

    def show_message(stdscr, title: str, lines: List[str], footer: str = "Press any key...") -> None:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        title_line = f" {title} "
        stdscr.addstr(0, max(0, (w - len(title_line)) // 2), title_line, curses.A_BOLD)
        y = 2
        for line in lines:
            if y >= h - 2:
                break
            stdscr.addstr(y, 2, line[: max(0, w - 4)])
            y += 1
        stdscr.addstr(h - 1, 2, footer[: max(0, w - 4)], curses.A_DIM)
        stdscr.refresh()
        stdscr.getch()

    def flow_convert_folder(stdscr) -> None:
        ensure_dirs()
        rvt_to_ifc = converter_script_path()

        folders = list_subfolders(RVT_ROOT)
        if not folders:
            show_message(stdscr, "Error", [f"No subfolders in {RVT_ROOT}."])
            return

        idx = pick_from_list_curses(
            stdscr,
            f"Pick a folder in {RVT_ROOT}",
            [p.name for p in folders],
            footer="↑/↓ Enter  Esc back",
        )
        if idx is None:
            return

        folder = folders[idx]
        files = list_rvt_files(folder)
        if not files:
            show_message(stdscr, "No files", [f"No .rvt files in {folder.name}"])
            return

        out_dir = IFC_ROOT / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)

        ok_count = 0
        fail_count = 0

        for i, in_rvt in enumerate(files, start=1):
            out_ifc = out_dir / (in_rvt.stem + ".ifc")
            manifest = out_dir / (in_rvt.stem + "_manifest.json")
            log_path = out_dir / (in_rvt.stem + ".log")

            def progress_cb(tail_lines: List[str]) -> None:
                stdscr.clear()
                h, w = stdscr.getmaxyx()
                stdscr.addstr(0, 2, f"Folder: {folder.name}  [{i}/{len(files)}]", curses.A_BOLD)
                stdscr.addstr(2, 2, f"File: {in_rvt.name}")
                stdscr.addstr(3, 2, f"Out:  {out_ifc.name}")
                stdscr.addstr(4, 2, f"Log:  {log_path.name}")
                stdscr.addstr(5, 2, f"Args: --region {REGION} --ifcver {IFCVER} --save-manifest", curses.A_DIM)

                y = 7
                stdscr.addstr(y, 2, "Last log lines:", curses.A_UNDERLINE)
                y += 1
                for ln in tail_lines[-TAIL_LINES:]:
                    if y >= h - 2:
                        break
                    stdscr.addstr(y, 2, ln[: max(0, w - 4)])
                    y += 1

                stdscr.addstr(
                    h - 1,
                    2,
                    "Ctrl+C to stop.",
                    curses.A_DIM,
                )
                stdscr.refresh()

            try:
                ok, code = run_convert_one_logged(
                    rvt_to_ifc=rvt_to_ifc,
                    in_rvt=in_rvt,
                    out_ifc=out_ifc,
                    manifest_path=manifest,
                    log_path=log_path,
                    show_progress_cb=progress_cb,
                )
            except KeyboardInterrupt:
                show_message(
                    stdscr,
                    "Stopped",
                    [
                        "Conversion stopped by user (Ctrl+C).",
                        f"Output: {out_dir}",
                        "Logs are already saved for processed files.",
                    ],
                )
                return

            if ok:
                ok_count += 1
            else:
                fail_count += 1
                tail = tail_file(log_path, 12)
                show_message(
                    stdscr,
                    "Error",
                    [
                        f"File: {in_rvt.name}",
                        f"Exit code: {code}",
                        f"Log: {log_path}",
                        "",
                        *tail,
                    ],
                    footer="Press any key to continue...",
                )

        show_message(
            stdscr,
            "Done",
            [
                f"Folder: {folder.name}",
                f"Results: {out_dir}",
                "",
                "Logs: *.log (per file)",
                "Manifest: *_manifest.json (per file)",
            ],
        )

    def flow_convert_single(stdscr) -> None:
        ensure_dirs()
        rvt_to_ifc = converter_script_path()

        folders = list_subfolders(RVT_ROOT)
        if not folders:
            show_message(stdscr, "Error", [f"No subfolders in {RVT_ROOT}."])
            return

        idx = pick_from_list_curses(stdscr, f"Pick a folder in {RVT_ROOT}", [p.name for p in folders])
        if idx is None:
            return
        folder = folders[idx]

        files = list_rvt_files(folder)
        if not files:
            show_message(stdscr, "No files", [f"No .rvt files in {folder.name}"])
            return

        fidx = pick_from_list_curses(stdscr, f"Pick a .rvt from {folder.name}", [p.name for p in files])
        if fidx is None:
            return
        in_rvt = files[fidx]

        out_dir = IFC_ROOT / folder.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_ifc = out_dir / (in_rvt.stem + ".ifc")
        manifest = out_dir / (in_rvt.stem + "_manifest.json")
        log_path = out_dir / (in_rvt.stem + ".log")

        def progress_cb(tail_lines: List[str]) -> None:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            stdscr.addstr(0, 2, "Single file conversion", curses.A_BOLD)
            stdscr.addstr(2, 2, f"File: {in_rvt.name}")
            stdscr.addstr(3, 2, f"Out:  {out_ifc.name}")
            stdscr.addstr(4, 2, f"Log:  {log_path.name}")
            stdscr.addstr(5, 2, f"Args: --region {REGION} --ifcver {IFCVER} --save-manifest", curses.A_DIM)

            y = 7
            stdscr.addstr(y, 2, "Last log lines:", curses.A_UNDERLINE)
            y += 1
            for ln in tail_lines[-TAIL_LINES:]:
                if y >= h - 2:
                    break
                stdscr.addstr(y, 2, ln[: max(0, w - 4)])
                y += 1

            stdscr.addstr(h - 1, 2, "Ctrl+C to stop.", curses.A_DIM)
            stdscr.refresh()

        try:
            ok, code = run_convert_one_logged(
                rvt_to_ifc=rvt_to_ifc,
                in_rvt=in_rvt,
                out_ifc=out_ifc,
                manifest_path=manifest,
                log_path=log_path,
                show_progress_cb=progress_cb,
            )
        except KeyboardInterrupt:
            show_message(
                stdscr,
                "Stopped",
                [
                    "Stopped by user (Ctrl+C).",
                    f"Log: {log_path}",
                ],
            )
            return

        if ok:
            show_message(
                stdscr,
                "Done",
                [
                    f"IFC: {out_ifc}",
                    f"Manifest: {manifest}",
                    f"Log: {log_path}",
                ],
            )
        else:
            tail = tail_file(log_path, 12)
            show_message(
                stdscr,
                "Error",
                [
                    f"Exit code: {code}",
                    f"Log: {log_path}",
                    "",
                    *tail,
                ],
            )

    def main_curses(stdscr) -> None:
        curses.curs_set(0)
        stdscr.keypad(True)

        try:
            ensure_dirs()
        except Exception as e:
            show_message(stdscr, "Error", [str(e)])
            return

        items = [
            "Convert ONE file (sequential, with a log)",
            "Convert a FOLDER (sequential, with logs)",
            "Exit",
        ]

        while True:
            choice = pick_from_list_curses(
                stdscr,
                "RVT -> IFC menu",
                items,
                footer=f"↑/↓ Enter  Esc=exit | RVT:{RVT_ROOT} | IFC:{IFC_ROOT}",
            )
            if choice is None or choice == 2:
                return
            if choice == 0:
                flow_convert_single(stdscr)
            elif choice == 1:
                flow_convert_folder(stdscr)

    os.environ.setdefault("ESCDELAY", "25")
    import curses
    curses.wrapper(main_curses)


if __name__ == "__main__":
    main()
