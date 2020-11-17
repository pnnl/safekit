.. safekit documentation master file, created by
   sphinx-quickstart on Thu Jan  5 17:42:22 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. papers

.. _Deep Learning for Unsupervised Insider Threat Detection in Structured Cybersecurity Data Streams: https://aaai.org/ocs/index.php/WS/AAAIW17/paper/viewFile/15126/14668
.. _Recurrent Neural Network Language Models for Open Vocabulary Event-Level Cyber Anomaly Detection: https://arxiv.org/abs/1712.00557


Authors
=======

- Aaron Tuor (aaron.tuor@pnnl.gov)
- Ryan Baerwolf (rdbaerwolf@gmail.com)
- Robin Cosbey (rcosbey@live.com)
- Nick Knowles (knowles.nick@gmail.com)
- Elliot Skomski (elliottskomski@gmail.com)
- Sam Kaplan (samuelpkaplan@gmail.com)
- Brian Hutchinson (brian.hutchinson@wwu.edu)
- Nicole Nichols (nicole.nichols@pnnl.gov)
- Sean Robinson (greatblondelf@gmail.com)
- Rob Jasper (robert.jasper@pnnl.gov)

About Safekit
=============
Safekit is a python software package for anomaly detection from multivariate streams,
developed for the **AIMSAFE** (Analysis in Motion Stream Adaptive Foraging for Evidence) project at Pacific Northwest National Laboratory.
An exposition of the models in this package can be found in the papers:

- `Deep Learning for Unsupervised Insider Threat Detection in Structured Cybersecurity Data Streams`_
- `Recurrent Neural Network Language Models for Open Vocabulary Event-Level Cyber Anomaly Detection`_


The code of the toolkit is written in python using the tensorflow deep learning
toolkit and numpy.

Dependencies
============

Dependencies required for installation:

- Tensorflow 1.0 or above
- Numpy
- Scipy
- Sklearn
- Matplotlib

Python Distribution
-------------------

Safekit is written in python 2. Most functionality should be forwards compatible.


Installation
-------------

A virtual environment is recommended for installation. Make sure that tensorflow 1.0+ is installed in your virtual environment.

`Install tensorflow`_

From the terminal in your activated virtual environment:
Follow instructions to install TF:
https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/g3doc/get_started/os_setup.md

conda env

.. code-block::

   $ conda create -n safekit python=2.7
   $ (safekit) source activate safekit
   # CPU
   (safekit) $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp27-none-linux_x86_64.whl
   # GPU
   (safekit) $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp27-none-linux_x86_64.whl
   (safekit) $ pip install --ignore-installed --upgrade $TF_BINARY_URL
   (safekit) $ git clone https:/github.com/pnnl/safekit.git
   (safekit)) $ cd safekit/
   (safekit) $ python setup.py develop

To test your installation, from the top level directory run:

.. code-block:: bash

    $ tar -xjvf data_examples.tar.bz2
    $ python test/agg_tests.py data_examples/lanl/agg_feats data_examples/cert/agg_feats test.log
    $ python test/lanl_lm_tests.py data_examples/lanl/lm_feats/ test.log

These two tests should take about 10 to 15 minutes each depending on the processing capability of your system.
The tests range over many different model configurations and can be used as a somewhat comprehensive tutorial on the functionality of the code base.

.. _Install tensorflow: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html

Documentation
--------------
`Github hosted documentation`_

.. _Github hosted documentation: https://pnnl.github.io/safekit/
Docs can be read locally from the cloned repo by opening safekit/docs/_build/html/index.html with a browser.

