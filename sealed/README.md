# Sealed validation seeds

`seeds.json` is a stable repository-level case-selection manifest. The affine
description expands to 2,048 base seeds and is mixed with each mapped task ID by
the protected judge. Its namespace is separate from development, regression,
and fresh-fuzz schedules.

The values are deliberately inspectable; “sealed” means independent and
integrity-protected, not secret. Task workers must not edit or refresh this file.
After an intentional reviewed judge change, a maintainer may regenerate the
integrity manifest with `python -m neurogolf.cli integrity refresh --reviewed`.

