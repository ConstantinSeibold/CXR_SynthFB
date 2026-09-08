# Cut-paste pool (empty by default)

The optional cut-paste augmentation pastes real device photographs (RGBA cutouts) onto the
base CXRs, in addition to the procedurally rendered categories. The pool we used internally
was extracted from copyrighted source material and is not redistributable, so this directory
ships empty. The pipeline detects the empty pool and runs fully procedural — all 119
categories still render.

To supply your own pool, drop RGBA PNGs (transparent background) here:

```
cutouts/labels/<category>/*.png
```

Directories are auto-discovered at startup (`load_cutouts` in `SynthFB.py`); each directory
name becomes a COCO category.

## Optional manifest

By default a pasted cutout is annotated with its directory name. To map individual cutouts
onto canonical SynthFB classes instead (so e.g. a pacemaker photo is labeled
`pacemaker_body` rather than `my_pacemakers`), place a `book_manifest.json` next to this
directory (i.e. at `cutouts/book_manifest.json`) — a JSON list with one entry per PNG:

```json
[
  {
    "rel_path": "labels/my_pacemakers/img_001.png",
    "synthfb_category": "pacemaker_body"
  }
]
```

`rel_path` is relative to `cutouts/`. Cutouts without a manifest entry keep the generic
directory-name label. Valid `synthfb_category` values are the category names printed by the
generator at startup.
