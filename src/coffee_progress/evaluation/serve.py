"""Open a generated report in a browser with its video assets served locally."""

import argparse
import functools
import re
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReportHandler(SimpleHTTPRequestHandler):
    """Support byte ranges: browsers need partial MP4 reads for reliable seeking."""

    def send_head(self):
        self._remaining = None
        path = Path(self.translate_path(self.path))
        requested = self.headers.get("Range")
        if requested is None or path.suffix.lower() != ".mp4" or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
        if match and any(match.groups()):
            left, right = match.groups()
            start = int(left) if left else max(0, size - int(right))
            end = min(int(right), size - 1) if left and right else size - 1
            if 0 <= start <= end < size:
                handle = path.open("rb")
                handle.seek(start)
                self._remaining = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(self._remaining))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                return handle
        self.send_response(416)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return None

    def copyfile(self, source, outputfile):
        if self._remaining is None:
            return super().copyfile(source, outputfile)
        while self._remaining > 0:
            block = source.read(min(self._remaining, 64 * 1024))
            if not block:
                break
            outputfile.write(block)
            self._remaining -= len(block)


def create_server(report: str | Path, port: int = 8765) -> ThreadingHTTPServer:
    folder = Path(report).resolve()
    if folder.is_file():
        folder = folder.parent
    if not (folder / "index.html").is_file() and (folder / "qualitative/index.html").is_file():
        folder = folder / "qualitative"
    if not (folder / "index.html").is_file():
        raise FileNotFoundError(f"No index.html in {folder}")
    if not (folder / "videos").is_dir():
        raise FileNotFoundError(f"No videos folder in {folder}; keep the full report together")
    handler = functools.partial(ReportHandler, directory=str(folder))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Report directory or index.html path")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-open", action="store_true", help="Print URL without opening a browser"
    )
    args = parser.parse_args()
    try:
        server = create_server(args.report, args.port)
    except (OSError, ValueError) as exc:
        parser.error(f"{exc}. If the port is busy, choose --port 8766.")
    with server:
        url = f"http://127.0.0.1:{server.server_port}/index.html"
        print(f"Viewer: {url}\nKeep this terminal running. Press Ctrl+C to stop.", flush=True)
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
