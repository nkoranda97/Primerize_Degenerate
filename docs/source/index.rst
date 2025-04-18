``Primerize`` Documentation
===========================================

.. image:: _static/primerize_logo.png
    :align: right
    :alt: Primerize Logo
    :target: https://primerize.stanford.edu/
    :scale: 75 %


``Primerize`` (previously named ``NA_thermo``), is an archive of *Python* and *MATLAB* scripts for primer design and nucleic acid thermodynamic scripts developed by the `Das Lab <https://daslab.stanford.edu/>`_ at Stanford University for high-throughput RNA synthesis and design.

The algorithm designs *forward* (sense strand) and *reverse* (anti-sense strand) primers that minimize the total length, and therefore the total synthesis cost, of the oligonucleotides. Although developed independently, ``Primerize`` is a special case of the general ‘*Gapped Oligo Design*’ algorithm, optimizing the mispriming score and sequence span instead of T\ :sub:`m`.

An online user-friendly GUI is available as the `Primerize Server <https://primerize.stanford.edu/>`_.

The code for Primerize is available at `<https://github.com/ribokit/Primerize>`_`

----------

Table of Contents
----------------------------

.. toctree::
   :glob:
   :maxdepth: 2

   install
   example
   content
   matlab

----------

Experimental Protocol
----------------------------

Please visit the **Primerize** server for a complete guidance on primer designs and RNA synthesis:

* `DNA template design <https://primerize.stanford.edu/protocol/#temp_design>`_

* `IDT oligo ordering <https://primerize.stanford.edu/protocol/#IDT>`_

* `PCR assembly <https://primerize.stanford.edu/protocol/#PCR>`_

* `in vitro transcription <https://primerize.stanford.edu/protocol/#TX>`_

* `Massively parallel preparation <https://primerize.stanford.edu/protocol/#par_prep>`_

----------

License
----------------------------

Copyright &copy; of **Primerize** *Source Code* is described in `LICENSE <https://github.com/ribokit/Primerize/blob/master/LICENSE>`_.


----------

Reference
----------------------------

| Tian, S., Yesselman, J.D., Cordero, P., and Das, R. (**2015**)
| `Primerize: Automated Primer Assembly for Transcribing Interesting RNAs. <http://nar.oxfordjournals.org/content/43/W1/W522.full>`_
| *Nucleic Acid Research* **43 (W1)**: W522-W526.
|
| Tian, S., and Das, R. (**2017**)
| `Primerize-2D: Automated Primer Design for RNA Multidimensional Chemical Mapping. <https://academic.oup.com/bioinformatics/article-abstract/33/9/1405/2801460/Primerize-2D-automated-primer-design-for-RNA>`_
| *Bioinformatics* **33 (9)**: 1405-1406.

----------

Workflows
----------------------------

* `I have just discovered an RNA molecule </workflows/from_scratch/>`_

* `I want to learn an RNA’s secondary structure </workflows/2D_modeling/>`_

* `I want to test an RNA secondary structure model </workflows/mutation_rescue/>`_

* `I think my RNA has interesting alternative states </workflows/alternative_states/>`_


----------

Developed by **Das lab**, `Leland Stanford Junior University`.

README by `t47 <https://t47.io/>`_, *January 2016*.
