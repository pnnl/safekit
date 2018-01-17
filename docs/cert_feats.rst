CERT Features
=============

Aggregate Features
------------------

CERT aggregate features are in the compressed file:

cert_aggregate_features.tar.bz2

They were made according to the specifications in the Paper **Deep Learning for Unsupervised Insider Threat Detection in Structured Cybersecurity Data Streams**.
These should be used in conjunction with dnn_agg.py and lstm_agg.py. These are pretty big so will take a while to uncompress.

Example usage:

.. code-block:: bash

    $ tar -xjvf safekit/safekit/features/cert/cert_aggregate_features.tar.bz2
    $ cd safekit/safekit/models/
    $ python dnn_agg.py safekit/safekit/features/cert/cert_aggregate_features.txt output.txt ../features/specs/agg/cert_all_in_all_out_agg.json -skipheader
    $ python dnn_agg.py safekit/safekit/features/cert/cert_aggregate_features.txt output.txt ../features/specs/agg/cert_all_in_all_out_agg.json -skipheader


Our legal department wants this included:

This material was prepared as an account of work sponsored by an agency of the United States Government.
Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees,
nor any jurisdiction or organization that has cooperated in the development of these materials,
makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness,
or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer,
or otherwise does not necessarily constitute or imply its endorsement, recommendation,
or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute.
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830