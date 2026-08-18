# Using KerNet or other retinal morphology data

KerNet contains 3-D fluorescence and segmentation datasets of keratin networks in epithelial cells, including RPE-related material. It can help provide realistic cell boundaries, texture, and axial morphology.

It does **not** directly provide A2E or FAD molecular concentration maps. The safe workflow is:

1. Obtain the dataset under its stated license.
2. Select a morphology or segmentation volume.
3. Resample it into a documented physical voxel size.
4. Use the morphology as a support mask or spatial prior.
5. Generate or measure fluorophore concentration fields separately.
6. Keep morphology uncertainty and molecular uncertainty separate.
7. Never describe a keratin channel as A2E/FAD ground truth.

A future loader can convert an exported NPY/NPZ array into the package's `(species, z, y, x)` convention. The core forward model already supports 3-D arrays.
