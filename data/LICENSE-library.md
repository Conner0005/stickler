# Licence of `data/library.parquet`

`data/library.parquet` is derived from **SurfPro**: Hödl, Hermans, Dankloff, Piruska, Huck,
Robinson, *SurfPro – a curated database and predictive model of experimental properties of
surfactants*, Digital Discovery 2025, 4, 1176–1187, doi:10.1039/D4DD00393D;
https://github.com/BigChemistry-RobotLab/SurfPro (commit 35b8906).

SurfPro is licensed **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/).
That licence allows redistribution with attribution, for non-commercial purposes, with adapted
material shared under the same licence. Accordingly, this derived file is shared under
CC BY-NC-SA 4.0 and may only be used non-commercially. Changes made: temperature filtering,
InChIKey deduplication, unit conversion to SI, σ and provenance columns, structural family tags,
and selection of 40 candidates (see docs/data.md). The raw SurfPro files are not redistributed
here; `wetopt library --rebuild` fetches them per docs/data.md.

`data/regulations.yaml` is an original hardcoded snapshot written for this project (not legal
advice).
