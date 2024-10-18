# Intro

This is Pantograph, an machine-to-machine interaction interface for Lean 4.
Its main purpose is to train and evaluate theorem proving agents. The main
features are:
1. Interfacing via the Python library, REPL, or C Library
2. A mixture of expression-based and tactic-based proofs
3. Pantograph has been designed with facilitating search in mind. A theorem
   proving agent can simultaneously explore multiple branches of the proof.
4. Handling of metavariable coupling
5. Reading/Adding symbols from the environment
6. Extraction of tactic training data
7. Support for drafting

## Design Rationale

The Lean 4 interface is not conducive to search. Readers familiar with Coq may
know that the Coq Serapi was superseded by CoqLSP. In the opinion of the
authors, this is a mistake. An interface conducive for human operators to write
proofs is often not an interface conductive to search.

LeanDojo has architectural limitations that prevent it from gaining new features
such as drafting without refactoring. Almost all of Pantograph's business logic
is written in Lean, and Pantograph achieves tighter coupling between the data
extraction and proof search components.
