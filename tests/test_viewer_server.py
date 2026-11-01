import threading
import urllib.request

import pytest

from coffee_progress.evaluation.serve import create_server


def test_server_serves_report_and_video_with_correct_type(tmp_path):
    (tmp_path / "index.html").write_text("<html>report</html>")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/test.mp4").write_bytes(b"video-bytes")
    with create_server(tmp_path / "index.html", port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(url + "/index.html") as response:
                assert b"report" in response.read()
            with urllib.request.urlopen(url + "/videos/test.mp4") as response:
                assert response.headers.get_content_type() == "video/mp4"
                assert response.read() == b"video-bytes"
            request = urllib.request.Request(
                url + "/videos/test.mp4", headers={"Range": "bytes=6-10"}
            )
            with urllib.request.urlopen(request) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == "bytes 6-10/11"
                assert response.read() == b"bytes"
        finally:
            server.shutdown()
            thread.join()


def test_missing_report_assets_fail_early(tmp_path):
    with pytest.raises(FileNotFoundError, match="index.html"):
        create_server(tmp_path, port=0)
    (tmp_path / "index.html").write_text("report")
    with pytest.raises(FileNotFoundError, match="videos"):
        create_server(tmp_path, port=0)


def test_run_root_automatically_selects_qualitative_report(tmp_path):
    viewer = tmp_path / "qualitative"
    viewer.mkdir()
    (viewer / "index.html").write_text("report")
    (viewer / "videos").mkdir()
    with create_server(tmp_path, port=0) as server:
        assert server.RequestHandlerClass.keywords["directory"] == str(viewer.resolve())
