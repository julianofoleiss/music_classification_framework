# Music Classification Framework

This repository contains general music classification utilities such as
feature extractors, aggregators, classification pipelines, voting systems,
texture selectors and other related utilities.

The idea is that general-purpose, reuse-friendly code is located in the 
*core* directory.

The *exp_gtzan_selframes* has experiment-specific code related to texture
selection. It is invoked by the mirex_sub.py script, which parses command-line
arguments. Please refer to it for more instructions.

*baseline_system.py* is a rewrite of *exp_gtzan_selframes*. It is easier to
understand, but somewhat less flexible.

This is Python 2.7 code. See requirements.txt for package requirements.

Any questions please email me! 

julianofoleiss@utfpr.edu.br

