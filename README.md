<p align="center">
  <img src="https://raw.githubusercontent.com/devojyoti96/P-AIRCARS/refs/heads/master/dark_logo.png" alt="P-AIRCARS Logo" width="200"/>
</p>
<p align="center">
  <h1>P-AIRCARS</h1> An automated spectropolarimetric calibration and imaging pipeline designed for solar radio observations using the <strong>Murchision Widefield Array (MWA)</strong> radio telescope. It performs end-to-end calibration, flagging, and imaging with a focus on dynamic solar data, supporting both spectral and temporal flexibility in imaging products.
</p>

## Background

<!-- start elevator-pitch -->

Solar radio data presents unique challenges due to the high variability and brightness of the Sun, as well as the need for high time-frequency resolution. The **P-AIRCARS** pipeline addresses these challenges by:

- Automating the calibration of interferometric data, including flux, phase, and polarization calibrations
- Supporting time-sliced and frequency-sliced imaging workflows
- Leveraging Dask for scalable parallel processing
- Providing hooks for integration with contextual data from other wavelegths for enhanced solar analysis

<!-- end elevator-pitch -->

## Documentation

P-AIRCARS documentation is available at: [paircars.readthedocs.io]

[paircars.readthedocs.io]: https://paircars.readthedocs.io 

## Quickstart

<!-- start quickstart -->

**MeerSOLAR** is distributed on [PyPI]. To use it:

1. Create conda environment with python 3.10

    ```text
    conda create -n paircars_env python=3.10
    conda activate paircars_env
    ```

2. Install P-AIRCARS in conda environment

   ```text
   pip install paircars
   ```

3. Initiate necessary metadata

    ```text
    init-paircars-setup --init
    ```
    
4. Run P-AIRCARS pipeline

    ```text
    run-mwa-paircars <path of target measurement set directory> <path of target metafits file> --workdir <path of work directory> --outdir <path of output products directory>
    ```    

That's all. You started P-AIRCARS pipeline for analysing your MWA solar observation 🎉.

5. To see all running P-AIRCARS jobs

    ```text
    show-paircars-status --show
    ```
    
6. To see prefect dashboard

   ```text
   run-mwa-mwalogger
   ```
      
7. To see local log of any job using the <jobid>

   ```text
   run-mwa-mwalogger --jobid <jobid>
   ```
   
7. Output products will be saved in : `<path of output products directory>`

[pypi]: https://pypi.org/project/paircars/

<!-- end quickstart -->

## Sample dataset
User can download and test entire P-AIRCARS pipeline using the sample dataset available in Zenodo: https://doi.org/10.5281/zenodo.16068485. Do not use this sample dataset for any publication without permission from the developer.


## Acknowledgements

P-AIRCARS is developed by Devojyoti Kansabanik (NCRA-TIFR, Pune, India and CPAESS-UCAR, Boulder, USA) and an incarnation of AIRCARS. If you use **P-AIRCARS** for analysing your MWA solar observations, include the following statement in your paper

```text
This MWA solar observations are analysed using P-AIRCARS pipeline.
```

1. P-AIRCARS software in zenodo: https://doi.org/10.5281/zenodo.16040507

<!-- will be updated one published.
and cite the following papers.


1. [First P-AIRCARS paper] [kansabanik2025]
[Kansabanik2025]: https://kansabanik-meersolar.org

2. [Second MeerSOLAR paper] [Patra2025]
[Patra2025]: https://patra-meersolar.org
-->

## License

This project is licensed under the MIT License.
