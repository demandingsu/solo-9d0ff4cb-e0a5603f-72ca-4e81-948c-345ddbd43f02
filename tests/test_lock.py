"""Locked-region ("patch" acceptance) behaviour.

A locked rectangle keeps the pixels and provenance of a published version,
inside the draft, through preview and publish.  The right/bottom bounds are
exclusive; at most one rectangle can be locked at a time.
"""

from __future__ import annotations

import io
import re
import threading

from PIL import Image

from app import main as main_module
from tests.conftest import fetch_csv, fetch_png, mask_bytes, png_bytes


def _post_publish(client):
    response = client.post("/publish", follow_redirects=False)
    assert response.status_code == 303, response.text
    return response


def _latest(client):
    versions = client.get("/versions").json()
    assert versions
    return versions[0]


def _lock(client, version_id, x0, y0, x1, y1):
    return client.post(
        "/lock",
        data={
            "version_id": version_id,
            "x0": str(x0),
            "y0": str(y0),
            "x1": str(x1),
            "y1": str(y1),
        },
        follow_redirects=False,
    )


def _publish_download(client):
    _post_publish(client)
    latest = _latest(client)
    png = client.get(f"/download/{latest['id']}/mosaic.png")
    csv = client.get(f"/download/{latest['id']}/sources.csv")
    assert png.status_code == 200 and csv.status_code == 200
    return latest, png, csv


def test_lock_keeps_old_pixels_and_provenance_inside_rectangle(
    client, upload, canvas
):
    canvas(4, 4)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=4, height=4)
    first, _, _ = _publish_download(client)

    # The draft moves on: a newer tile repaints everything.
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=4, height=4)

    # Lock the top-left 2x2. Right/bottom are exclusive, so (1,1) is inside
    # and (2,0)/(0,2) are outside.
    response = _lock(client, first["id"], 0, 0, 2, 2)
    assert response.status_code == 303
    assert "msg=" in response.headers["location"]

    rgba = fetch_png(client).convert("RGBA")
    assert rgba.getpixel((0, 0))[:3] == (255, 0, 0)
    assert rgba.getpixel((1, 1))[:3] == (255, 0, 0)
    assert rgba.getpixel((2, 0))[:3] == (0, 255, 0)
    assert rgba.getpixel((2, 2))[:3] == (0, 255, 0)

    rows = fetch_csv(client)
    assert rows[1 + 0] == ["0", "0", "A", "1"]
    assert rows[1 + 1] == ["1", "0", "A", "1"]
    assert rows[1 + 4] == ["0", "1", "A", "1"]
    assert rows[1 + 2] == ["2", "0", "B", "1"]
    assert rows[1 + 8] == ["0", "2", "B", "1"]

    # Publish the locked draft and check the exported files agree.
    latest, png, csv = _publish_download(client)
    image = Image.open(io.BytesIO(png.content)).convert("RGBA")
    assert image.getpixel((1, 1))[:3] == (255, 0, 0)
    assert image.getpixel((2, 1))[:3] == (0, 255, 0)
    text = csv.content.decode("utf-8").splitlines()
    assert "0,0,A,1" in text
    assert "2,1,B,1" in text
    manifest = client.get(f"/download/{latest['id']}/manifest.json").json()
    assert manifest["lock"]["seq"] == first["seq"]
    assert manifest["lock"]["rect"] == {"x0": 0, "y0": 0, "x1": 2, "y1": 2}
    assert {"tile_id": "A", "revision": 1} in [
        {k: s[k] for k in ("tile_id", "revision")} for s in manifest["lock"]["sources"]
    ]


def test_locked_transparent_pixel_keeps_empty_source(client, upload, canvas):
    canvas(2, 2)
    # Published version A has a cloud pixel at (0, 0) and nothing else there.
    upload(
        tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2,
        cloud={(0, 0)},
    )
    first, _, _ = _publish_download(client)

    # The draft fills the cloudy pixel with a newer clear tile.
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=2, height=2)
    _lock(client, first["id"], 0, 0, 2, 2)

    rgba = fetch_png(client).convert("RGBA")
    assert rgba.getpixel((0, 0))[3] == 0  # locked transparency wins
    assert rgba.getpixel((1, 1))[:3] == (255, 0, 0)
    rows = fetch_csv(client)
    assert rows[1] == ["0", "0", "", ""]
    assert "B," not in "\n".join(",".join(row) for row in rows[1:])


def test_lock_requires_same_canvas_size(client, upload, canvas):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, _, _ = _publish_download(client)

    canvas(3, 3)
    response = _lock(client, first["id"], 0, 0, 2, 2)
    assert response.status_code == 303
    assert "err=" in response.headers["location"]
    # No lock was recorded.
    assert "补片锁区" in client.get("/").text


def test_canvas_dimensions_frozen_while_locked(client, upload, canvas):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, _, _ = _publish_download(client)
    assert _lock(client, first["id"], 0, 0, 1, 1).status_code == 303

    changed = canvas(3, 3)
    assert changed.status_code == 303
    assert "err=" in changed.headers["location"]
    assert fetch_png(client).size == (2, 2)

    # Same size is a no-op and still allowed.
    assert canvas(2, 2).status_code == 303


def test_unlock_and_relock_restore_draft_composition(client, upload, canvas):
    canvas(4, 4)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=4, height=4)
    first, _, _ = _publish_download(client)
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=4, height=4)

    _lock(client, first["id"], 0, 0, 2, 2)
    assert fetch_png(client).convert("RGBA").getpixel((0, 0))[:3] == (255, 0, 0)

    unlocked = client.post("/unlock", follow_redirects=False)
    assert unlocked.status_code == 303
    assert "msg=" in unlocked.headers["location"]
    assert fetch_png(client).convert("RGBA").getpixel((0, 0))[:3] == (0, 255, 0)

    # Re-lock a different rectangle (bottom-right) of the same version.
    assert _lock(client, first["id"], 2, 2, 4, 4).status_code == 303
    rgba = fetch_png(client).convert("RGBA")
    assert rgba.getpixel((3, 3))[:3] == (255, 0, 0)
    assert rgba.getpixel((0, 0))[:3] == (0, 255, 0)

    # Publishing after the re-lock freezes the new rectangle, not the old one.
    latest, png, csv = _publish_download(client)
    image = Image.open(io.BytesIO(png.content)).convert("RGBA")
    assert image.getpixel((3, 3))[:3] == (255, 0, 0)
    assert image.getpixel((0, 0))[:3] == (0, 255, 0)
    manifest = client.get(f"/download/{latest['id']}/manifest.json").json()
    assert manifest["lock"]["rect"] == {"x0": 2, "y0": 2, "x1": 4, "y1": 4}
    text = csv.content.decode("utf-8").splitlines()
    assert "3,3,A,1" in text
    assert "0,0,B,1" in text


def test_locked_rectangle_is_not_moved_by_later_lock_during_publish(
    client, upload, canvas, monkeypatch
):
    """A publish renders whichever lock was frozen in its snapshot."""
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, _, _ = _publish_download(client)
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=2, height=2)
    assert _lock(client, first["id"], 0, 0, 1, 1).status_code == 303

    real_render = main_module.render_mosaic
    fired = threading.Event()
    injected = False

    def render_then_relock(*args, **kwargs):
        nonlocal injected
        if injected:
            # A nested preview refresh triggered by the lock change renders
            # normally; only the publish render does the mutation.
            return real_render(*args, **kwargs)
        injected = True
        fired.set()
        client.post("/unlock", follow_redirects=False)
        client.post(
            "/lock",
            data={
                "version_id": first["id"],
                "x0": "1", "y0": "1", "x1": "2", "y1": "2",
            },
            follow_redirects=False,
        )
        return real_render(*args, **kwargs)

    monkeypatch.setattr(main_module, "render_mosaic", render_then_relock)
    assert client.post("/publish", follow_redirects=False).status_code == 303
    assert fired.is_set()

    latest = _latest(client)
    png = client.get(f"/download/{latest['id']}/mosaic.png")
    csv = client.get(f"/download/{latest['id']}/sources.csv")
    image = Image.open(io.BytesIO(png.content)).convert("RGBA")
    # Snapshot froze the top-left lock; the later unlock/relock cannot leak in.
    assert image.getpixel((0, 0))[:3] == (255, 0, 0)
    assert image.getpixel((1, 1))[:3] == (0, 255, 0)
    manifest = client.get(f"/download/{latest['id']}/manifest.json").json()
    assert manifest["lock"]["rect"] == {"x0": 0, "y0": 0, "x1": 1, "y1": 1}
    text = csv.content.decode("utf-8").splitlines()
    assert "0,0,A,1" in text
    assert "1,1,B,1" in text


def test_lock_rectangle_validation(client, upload, canvas):
    canvas(3, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=3, height=2)
    first, _, _ = _publish_download(client)

    for rect in [(0, 0, 0, 1), (0, 0, 1, 0), (-1, 0, 1, 1),
                 (0, 0, 4, 1), (0, 0, 1, 3), (3, 0, 3, 1)]:
        response = _lock(client, first["id"], *rect)
        assert response.status_code == 303
        assert "err=" in response.headers["location"], rect

    assert _lock(client, first["id"], 0, 0, 3, 2).status_code == 303


def test_published_lock_manifest_includes_withdrawn_source(
    client, upload, canvas
):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, _, _ = _publish_download(client)
    upload(tile_id="A", revision=2, colour=(0, 0, 255), width=2, height=2)

    # Lock version #1, then withdraw the revision it used (r1 of A).
    assert _lock(client, first["id"], 0, 0, 2, 2).status_code == 303
    page = client.get("/").text
    rid = re.search(
        r"<td class=\"mono\">A</td>\s*<td class=\"num\">r1</td>.*?/tiles/(\d+)/withdraw",
        page,
        re.S,
    ).group(1)
    client.post(f"/tiles/{rid}/withdraw", follow_redirects=False)

    rgba = fetch_png(client).convert("RGBA")
    assert rgba.getpixel((0, 0))[:3] == (255, 0, 0)  # locked r1 survives
    latest, _, csv = _publish_download(client)
    manifest = client.get(f"/download/{latest['id']}/manifest.json").json()
    asserted = [
        s for s in manifest["lock"]["sources"]
        if s["tile_id"] == "A" and s["revision"] == 1
    ]
    assert asserted and asserted[0]["withdrawn"] is True
    assert asserted[0]["image_sha256"]
    assert "A,1" in csv.content.decode("utf-8")


def test_missing_locked_file_aborts_publish_and_keeps_downloads(
    client, upload, canvas
):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, png1, csv1 = _publish_download(client)
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=2, height=2)
    assert _lock(client, first["id"], 0, 0, 2, 2).status_code == 303
    latest, png2, csv2 = _publish_download(client)
    assert latest["id"] != first["id"]

    # Remove the locked version's PNG behind the app's back.
    data_dir = client.app.state.runtime.data_dir
    version_dir = data_dir / "published" / first["id"]
    # The directory name includes the sequence prefix; locate it.
    version_dir = next((data_dir / "published").glob(f"*{first['id']}"))
    (version_dir / "mosaic.png").unlink()

    publish = client.post("/publish", follow_redirects=False)
    assert publish.status_code == 303
    assert "err=" in publish.headers["location"]
    versions = client.get("/versions").json()
    assert len(versions) == 2  # no half-published version was added
    assert client.get("/download/latest/mosaic.png").content == png2.content
    assert client.get("/download/latest/sources.csv").content == csv2.content
    # The old version's CSV download is also untouched.
    assert client.get(f"/download/{first['id']}/sources.csv").content == csv1.content

    # Preview generation also fails, leaving the previous preview in place.
    response = client.get("/preview.png")
    assert response.status_code == 200
    assert "锁定版本" in client.get("/").text


def test_missing_locked_file_aborts_preview_and_reports_error(
    client, upload, canvas
):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, _, _ = _publish_download(client)
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=2, height=2)
    assert _lock(client, first["id"], 0, 0, 2, 2).status_code == 303
    preview = fetch_png(client)

    version_dir = next(
        (client.app.state.runtime.data_dir / "published").glob(f"*{first['id']}")
    )
    (version_dir / "sources.csv").unlink()

    # Any draft change triggers regeneration, which must fail closed.
    upload(tile_id="C", revision=1, colour=(0, 0, 255), width=1, height=1)
    assert client.get("/preview.png").status_code in (404, 409, 503)
    csv_response = client.get("/preview.csv")
    assert csv_response.status_code != 200
    page = client.get("/").text
    assert "锁定版本" in page
    # The previously generated preview dir is untouched on disk.
    assert preview.size == (2, 2)


def test_lock_and_publish_are_frozen_against_concurrent_changes(
    client, upload, canvas, monkeypatch
):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, _, _ = _publish_download(client)
    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=2, height=2)
    assert _lock(client, first["id"], 0, 0, 1, 1).status_code == 303

    real_render = main_module.render_mosaic
    fired = threading.Event()

    def slow_render(*args, **kwargs):
        # While this publish renders, move both the draft and the lock.
        fired.set()
        client.post("/unlock", follow_redirects=False)
        client.post(
            "/tiles",
            files={
                "image": ("t.png", png_bytes(2, 2, (0, 0, 255)), "image/png"),
                "mask": ("m.png", mask_bytes(2, 2), "image/png"),
                "tile_id": (None, "C"),
                "revision": (None, "1"),
                "x": (None, "0"),
                "y": (None, "0"),
            },
            follow_redirects=False,
        )
        return real_render(*args, **kwargs)

    monkeypatch.setattr(main_module, "render_mosaic", slow_render)
    assert client.post("/publish", follow_redirects=False).status_code == 303
    assert fired.is_set()

    latest = _latest(client)
    png = client.get(f"/download/{latest['id']}/mosaic.png")
    csv = client.get(f"/download/{latest['id']}/sources.csv")
    image = Image.open(io.BytesIO(png.content)).convert("RGBA")
    # The snapshot was taken before the unlock, so the lock still applied and
    # no later upload leaked in.
    assert image.getpixel((0, 0))[:3] == (255, 0, 0)
    assert image.getpixel((1, 1))[:3] == (0, 255, 0)
    text = csv.content.decode("utf-8")
    assert "0,0,A,1" in text
    assert "C,1" not in text
    manifest = client.get(f"/download/{latest['id']}/manifest.json").json()
    assert manifest["lock"]["seq"] == first["seq"]


def test_mismatched_locked_canvas_is_reported_and_aborts_generation(
    client, upload, canvas
):
    canvas(2, 2)
    upload(tile_id="A", revision=1, colour=(255, 0, 0), width=2, height=2)
    first, _, _ = _publish_download(client)
    assert _lock(client, first["id"], 0, 0, 2, 2).status_code == 303

    # Simulate a hand-edited database: the stored lock claims a 3x3 canvas
    # while the draft is 2x2.  Generation must refuse rather than clamp.
    data_dir = client.app.state.runtime.data_dir
    import sqlite3

    from app import mosaic

    conn = sqlite3.connect(data_dir / "mosaic.sqlite3")
    lock = mosaic.RegionLock.from_json(
        conn.execute("SELECT value FROM settings WHERE key='lock_json'").fetchone()[0]
    )
    bad = mosaic.RegionLock(
        version_id=lock.version_id,
        seq=lock.seq,
        dir_rel=lock.dir_rel,
        x0=0,
        y0=0,
        x1=3,
        y1=3,
        canvas_width=3,
        canvas_height=3,
        fingerprint=lock.fingerprint,
        png_sha256=lock.png_sha256,
        csv_sha256=lock.csv_sha256,
    )
    conn.execute(
        "UPDATE settings SET value=? WHERE key='lock_json'", (bad.to_json(),)
    )
    conn.commit()
    conn.close()

    upload(tile_id="B", revision=1, colour=(0, 255, 0), width=2, height=2)
    page = client.get("/").text
    assert "与当前画幅不一致" in page
    assert client.get("/preview.png").status_code in (404, 409, 503)
