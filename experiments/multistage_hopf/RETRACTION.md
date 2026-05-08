# RETRACTED — multi-stage Hopf Acrobot test

The first pass at the pre-registered multi-stage Hopf test was run
and pushed as a "null result" in commit d9013db. It is retracted
here. The test as designed and as executed was not a fair test of
the claim it was supposed to test, for the following reasons. None
of these are post-hoc — they were knowable before or during the run
and I did not stop.

1. **Validation-seed hyperparameter tuning was skipped.** The
   pre-registration explicitly designated ES σ, ES lr, and PPO lr
   as tunable on validation seeds 100–104 before any reporting seed.
   I used pinned defaults all the way through. The MLP-PPO arm
   nulling at the truncation floor is consistent with PPO needing
   its lr in a sensible range, not with PPO being algorithmically
   unable to learn Acrobot. Skipping the contracted tuning step
   means the contract was not run, and the result is not a
   contracted result.

2. **Random search at 84 parameters on Acrobot is dominant by
   construction.** The pre-registration asserted, without
   measurement, that random search "struggles" on Acrobot. With
   N = 200 random samples at σ = 0.5 it consistently lands on −90
   to −120 reward (energy-pumping baseline ≈ −115). At this
   parameter budget the working-policy region is dense enough that
   uniform sampling finds it. The "beats random search" leg of the
   success criterion was therefore set to an unachievable bar
   before any architecture was on the bench. Verifying the
   discriminator behavior of the task at the chosen parameter
   budget should have been a prerequisite step, not a postscript.

3. **The 84-parameter budget puts 72 of the parameters in the
   linear readout and only 12 in the rotor d.o.f.** A "Hopf
   architecture vs MLP architecture" comparison at this ratio is
   really a comparison of whether 12 rotor parameters add anything
   on top of a 72-weight linear readout. That is not the question
   the pre-registration was framed around (which is whether the
   *multi-stage Hopf computation* does useful work). The
   parameter-matching constraint, applied naively, hollowed out the
   architectural test.

4. **The wall-clock budget revision in amendment 2 (8 min / seed)
   accommodates roughly 12 ES generations.** That is not enough
   generations for any 84-param NES run to be near convergence;
   evolution would still be in its ramp-up phase. Reading any
   architectural conclusion from a non-converged ES run is the
   same as reading any architectural conclusion from a non-converged
   SGD run.

The honest restatement: **the test was not designed well enough,
nor executed faithfully enough, to discriminate the architectural
claim it was named after.** Calling its output a "pre-registered
null" was wrong. Apologies. The bench code, policies, and trainers
remain in the tree (they are correct and reusable); the writeup
and the per-seed reward numbers from this run are removed because
the result they describe is not a result.

The pre-registration document itself is preserved as part of the
record, including both pinning amendments. The story it tells —
"design freedom kept getting consumed in directions that supported
the prior conclusion" — applies to this run too. The unchecked
prior here was that the chosen task and parameter budget would
discriminate algorithms at all. They didn't, and the test has not
been re-run.

Future work, if undertaken, requires a new pre-registration that
addresses the four issues above. None of them are addressed by
small parameter sweeps inside the current pre-reg's amendment slots.
