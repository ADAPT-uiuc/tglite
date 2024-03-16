Getting Started
---------------

TGLite currently only support PyTorch as backend.

Installing on Linux
```````````````````

Prerequisites
^^^^^^^^^^^^^

python3.7 or later

gcc6.1 or later

pip

Installation
^^^^^^^^^^^^
.. code-block:: console

   $ pip install tglite

Verification
^^^^^^^^^^^^
To verify the installation, run the following in Python:

.. code-block:: python

   import tglite
   print(tglite.__version__)
   
Building from source
`````````````````````
To install the latest TGLite code for testing or development on the core, you will need to build TGlite from source.

Create and activate a python environment:

.. code-block:: console

   $ conda create -n tglite python=3.7
   $ conda activate tglite

Install dependencies that have CUDA versions:

.. code-block:: console
   
   $ pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
   $ pip install torch-scatter==2.1.0+pt112cu116 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

Get the TGLite source:

.. code-block:: console

   $ git clone https://github.com/ADAPT-uiuc/tglite.git
   $ cd tglite

Then install the package locally:

.. code-block:: console
   
   $ python -m pip install --upgrade build
   $ python -m pip install .

This will build the C++ extension (which requires C++14 and OpenMP), install
the rest of the dependencies (as listed in `pyproject.toml`), and then install
the `tglite` package.

Development Mode
^^^^^^^^^^^^^^^^

Development mode allows easily editing the code without having to re-install
the package. However, this only applies to the python code. When editing the
C++ extension code, it needs to be re-compiled again. Use `--editable` option of pip’s install sub-command:

.. code-block:: console

   $ python -m pip install --editable .


Running Tests
^^^^^^^^^^^^^

Unit tests are located in `tests` directory. First, install the testing
dependencies specified in `pyproject.toml`. Doing so might overwrite the dev
mode install, so you might need to re-enable dev mode. Then, exercise the tests
using the `pytest` utility.

.. code-block:: console
   
   # install test dependencies
   $ pip install '.[test]'

   # re-enable dev mode install
   $ pip uninstall -y tglite
   $ python -m pip install --editable .

   # run with test coverage report
   $ pytest --cov=tglite


Running Examples
^^^^^^^^^^^^^^^^
Inside the `examples <https://github.com/ADAPT-uiuc/tglite/tree/main/examples>`_ directory of the repository, several CTDG models have been implemented using `tglite`.
To run these example models, install the additional dependencies and download the datasets:

.. code-block:: console
   $ cd examples
   $ pip install -r requirements.txt # or "conda install -c conda-forge pandas scikit-learn" using conda
   $ ./download-data.sh
   $ python gen-data-files.py --data wiki-talk

This will download the datasets inside `examples/data/`, one can also download data to other places.

Use the scripts in `examples/exp` as a starting point, e.g.:

.. code-block:: console
   $ ./exp/tgat.sh --data-path . -d wiki --epochs 3



