---
title: 'SCEMATK: Single Cell Morphological Profiling in Whole Slide Images'
tags:
  - Python
  - digital pathology
  - morphological profiling
authors:
  - name: Hugh Warden
    orcid: 0000-0002-4308-7316
    corresponding: true
    affiliation: 1
  - name: Jesko Wagner
    orcid: 0000-0001-9805-7192
    affiliation: 1
  - name: Ava Khamseh
    orcid: 0000-0001-5203-2205
    corresponding: true
    affiliation: "1, 2"
affiliations:
 - name: MRC Human Genetics Unit, Institute of Genetics and Cancer, University of Edinburgh, Edinburgh, UK
   index: 1
 - name: School of Informatics, University of Edinburgh, Edinburgh, UK
   index: 2
date: 02 February 2025
bibliography: paper.bib
---

# Summary

Digital pathology is an important field of study for fast and accurate diagnostic processes as well as for structural analysis of diseased tissue. High resolution pathology images are often shared amongst pathologists and annotated. In recent years, there has been substantial work to apply machine learning to these images for investigative and diagnostic tasks such as disease status prediction. Due to the large size of the images, technical expertise and computational resources are often required to open and analyse these images at scale. SCEMATK (Single Cell Extraction and Morphological Analysis Tool-Kit) provides an efficient way to open and convert digital pathology images into a phenotypic data set of morphological features, rather than image pixel-level input, at single cell resolution, for performing interpretable/explainable machine learning tasks at scale. Furthermore, SCEMATK focuses on the uniform processing of patches, allowing for spatial analysis of the morphology of cells across patches to capture detailed information about local and global tissue structure. All of these processes can be run interactively in a Jupyter notebook, allowing for easy manipulation of images too large to fit into memory, as well as in parallel on High Performance Computing platforms. 

# Statement of Need

Due to the large size of digital pathology whole slide images (WSIs), substantial computational resources are required to load, process, and analyse these data. Existing open source tools typically do not focus primarily on the morphological profiling of these images at single cell resolution. QuPath provides comprehensive tools for image viewing, cell detection and manual annotation [@bankhead2017qupath]. QuPath can perform some morphological profiling however more complex analytical workflows require a steep learning curve. Other software focuses on preparing WSIs for image level machine learning using efficient tiling and patch extraction strategies [@pocock2022tiatoolbox; @pourakpour2025histomicstk; @dolezal2024slideflow; @barcellona2024slidetiler]. Although some of these tools support morphological profiling, the independent processing of tiles, rather than treating the image as a whole, can introduce inconsistencies when calculating spatial morphlogical statistics. Experiments using SCEMATK show that inclusion of spatial statitics can significantly increase performance in prediction tasks. This makes the consistent application of algorithms across an image vital for these calculations and why SCEMATK was made to focus on this in a performant manner.

SCEMATK addresses this gap by providing a unified, easily extensible interface for whole slide image preprocessing, object segmentation, and morphological profiling, implemented using lazy-loading strategies and applicable to any staining modality including H&E, IHC and multiplex. This design enables efficient analysis on standard personal computers while remaining scalable to high-performance computing environments. In contrast to existing morphological profiling tools, SCEMATK applies image transformations consistently across the entire slide prior to feature extraction of dozens of interpretable cellular features, allowing computation of spatial statistics that characterise the local distribution of phenotypes within each cell’s neighbourhood.

SCEMATK has been applied to a data set of WSI's of hepatocellular carcinoma in an in vivo model. It enabled researchers to extract morphological features and use these to train a machine learning classifier to identify cancerous cells, mutations and time from onset of cancer. The uniform processing of slides made possible the calculation of spatial statistics that greatly increased the accuracy of these predictions. The ability to use morphological profiles, rather than operating on image based machine learning, allowed for more robust interpretable machine learning techniques to be used to identify key changed in morphology driving the prediction

![Example outputs of `scematk`. (a) The image patch can be loaded and viewed with scale bar. (b) Normalisation can be applied with reference to a target image to reduce inter image stain variability. (c) Various segmentation strategies can be applied to identify regions of the tissue corresponding to (i) tissue, (ii) nuclei, (iii) cytoplasms and (iv) cells. (d) The image can be deconvolved to quantify the presence of H&E and IHC stains. (e) (i) Masks and images are passed to a morphological profiler to calculate dozens of interpretable morphological features and (ii)) these features can be viewed overlayed on the original image.](scematk_overview.png)

SCEMATK provides these basic tools for analysing images: 

- Image Viewing: Enables the user to view a thumbnail of an image or a specify a region to view in full resolution.
- Image Normalisation: Apply a colour transform to the image to fit that of a target image, combatting batch effects that may arise from differences in staining protocols between samples.
- Stain Deconvolution: Convert the image from RGB to a new colour space showing the intensities of different stains applied to the sample, indicating different biological properties of the tissue.
- Object Segmentation: Identify nuclei of cells within an image as well as an approximation of the cell and cytoplasm.
- Morphological Profiling: Convert the masks and deconvolved image into a table of morphological properties of each cell.
- Feature Plotting: Overlay morphological features on an image to view them over a tissue.

SCEMATK provides an all in one easy to use interface for carrying out single cell morphological profiling on digital pathology images with limited computational resources.

# Acknowledgements

HW and JW are supported by an MRC PhD studentship (grant no. MC_ST_00035). AK was supported by a Langmuir Talent Development Fellowship from the Institute of Genetics and Cancer, and a philanthropic donation from Hugh and Josseline Langmuir.

# References