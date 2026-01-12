:-include('lhprove.pro').

:- use_module(library(plunit)).

:- begin_tests(lhprove_equiv).

% Helper pattern used in every test:
%   lhprove/2 succeeds iff hprove/1 succeeds
%
% We encode the equivalence in one clause, so each test is a genuine “iff” test.

test(eq_identity_p) :-
  G = (p:-[p]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_k_like_drop_q) :-
  G = (p:-[q,p]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_chain_two_steps) :-
  % From p and (q:-[p]) and (r:-[q]) derive r
  G = (r:-[(r:-[q]), (q:-[p]), p]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_s_like) :-
  % From p, (q:-[p]) and (r:-[p,q]) derive r
  G = (r:-[(r:-[p,q]), (q:-[p]), p]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_three_premises_two_derived) :-
  % From p derive q and r; then use (t:-[p,q,r]) to derive t
  G = (t:-[(t:-[p,q,r]), (q:-[p]), (r:-[p]), p]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_two_independent_facts) :-
  % Just check multiple atomic assumptions are handled
  G = (u:-[u, v, u, v]),  % provable because u is an assumption
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_nested_premise_single_arg) :-
  % Uses nested Horn premise: (f:-[(g:-[x])]) together with x
  % Goal provides the clause for f and the fact x.
  G = (f:-[(f:-[(g:-[x])]), x]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_nested_premise_two_args) :-
  G = (h:-[(h:-[(g:-[x,y])]), x, y]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

% ---- Negative cases (still as equivalence tests) ----

test(eq_fail_empty_ctx) :-
  G = (p:-[]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_fail_missing_atom) :-
  G = (p:-[q]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_fail_missing_q) :-
  G = (r:-[(r:-[p,q]), p]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

test(eq_fail_nested_missing_y) :-
  G = (h:-[(h:-[(g:-[x,y])]), x]),
  ( hprove(G) -> lhprove(G,_)
  ; \+ lhprove(G,_)
  ).

:- end_tests(lhprove_equiv).
