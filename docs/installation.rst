.. highlight:: shell

============
Installation
============


Stable release
--------------

The maintained fork is intended to be installed from source so that the Python,
PyTorch, and `torch-npu` stack can be pinned to the local CANN environment.

For a CPU-only development environment:

.. code-block:: console

    $ python3.10 -m venv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements_dev.txt
    $ make install

For NPU runs, use the matching Makefile target for your architecture so the
selected `torch-npu` build stays aligned with the local Ascend software stack.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for pytorch-hccl-tests can be downloaded from the `Github repo`_.

You can either clone the maintained fork:

.. code-block:: console

    $ git clone https://github.com/huawei-csl/pytorch-hccl-tests.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/huawei-csl/pytorch-hccl-tests/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python -m pip install .


.. _Github repo: https://github.com/huawei-csl/pytorch-hccl-tests
.. _tarball: https://github.com/huawei-csl/pytorch-hccl-tests/tarball/master
