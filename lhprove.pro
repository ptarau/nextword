%% X is the proof term of T
%% or, via Curry-Howard, X:T holds when X is an inhabitant of the type T
lprove(T,X):-lprove(X,T,[]),!.

lprove(X,A,Vs):-memberchk(X:A,Vs),!. % leaf variable
lprove(l(X,E),(A->B),Vs):-!,lprove(E,B,[X:A|Vs]).  % introduce lambda term
    lprove(E,G,Vs1):-
    select(S:(A->B),Vs1,Vs2),       % source of application
    lprove_imp(T,A,B,Vs2),          % target of application
    !,
    lprove(E,G,[a(S,T):B|Vs2]).     % apply source to traget

lprove_imp(l(X,E),(C->D),B,Vs):-!,lprove(E,(C->D),[X:(D->B)|Vs]).
lprove_imp(E,A,_,Vs):-memberchk(E:A,Vs).

/*
?- T=((p->(p->q)->q)),lprove(T,Proof).                                               T = (p->(p->q)->q),
Proof = l(_A, l(_B, a(_B, _A))).


*/

%% Horn term representation of chained implications


hprove(A):-hprove(A,[]).

hprove(A,Vs):-memberchk(A,Vs),!.
hprove((B:-As),Vs1):-!,append(As,Vs1,Vs2),hprove(B,Vs2).
hprove(G,Vs1):-          % atomic(G), G not on Vs1
  memberchk((G:-_),Vs1), % if not, we just fail
  select((B:-As),Vs1,Vs2), % outer select loop
  select(A,As,Bs),         % inner select loop
  hprove_imp(A,B,Vs2), % A element of the body of B
  !,
  trimmed((B:-Bs),NewB), % trim off empty bodies
  hprove(G,[NewB|Vs2]).

hprove_imp((D:-Cs),B,Vs):- !,hprove((D:-Cs),[(B:-[D])|Vs]).
hprove_imp(A,_B,Vs):-memberchk(A,Vs).

trimmed((B:-[]),R):-!,R=B.
trimmed(BBs,BBs).

/*
?- H=(r:-[(r:-[p, q]), (q:-[p]), p]),hprove(H).
H = (r:-[(r:-[p, q]), (q:-[p]), p]).
*/


%% ------------------------------------------------------------------
%% lhprove/2 : proof-producing Horn-list prover (direct, no translation)
%%
%% Produces a closed lambda-term (using l/2 and a/2) witnessing provability
%% of Horn-list goals, in the same spirit as lprove/2.
%%
%% Design choice (to keep proofs aligned with list order / Herbrand arguments):
%%   when reducing a clause (B:-As) we always discharge the LEFTMOST premise
%%   As = [A|Bs].  This makes the procedure order-sensitive (like left-nested ->)
%%   and yields proof terms that closely follow curried application order.
%%
%% Assumptions in a goal (B:-As) become lambda-bound variables whose bodies
%% can be applied (via a/2) to discharge premises and derive heads.
%% ------------------------------------------------------------------

%% entry point
lhprove(A,Proof):- lhprove_env(A,[],Proof).

%% -----------------------------
%% lhprove_env/3 : prove with env
%% -----------------------------

%% leaf: A already available in the environment with witness Proof
lhprove_env(A,Env,Proof):- memberchk(Proof:A,Env),!.

%% goal clause: introduce lambdas for each assumption, then prove the head
lhprove_env((B:-As),Env1,Proof):-!,
  lhprove_bind_assumptions(As,Env1,Env2,Proof,BodyProof),
  lhprove_env(B,Env2,BodyProof).

%% atomic goal: saturate/reduce the Horn environment until goal becomes available
lhprove_env(G,Env1,Proof):-
  memberchk(_:(G:-_),Env1),          % must have some rule for G, else fail
  select(S:(B:-As),Env1,Env2),       % pick a clause to reduce (outer loop)
  As=[A|Bs],                         % discharge leftmost premise (order-sensitive)
  hprove_imp(A,B,Env2,T),            % obtain witness T for the premise A
  !,
  trimmed((B:-Bs),NewB),             % trim empty bodies
  lhprove_env(G,[a(S,T):NewB|Env2],Proof).

%% ---------------------------------------------------------
%% lhprove_bind_assumptions/5 : build lambdas + extend env
%% ---------------------------------------------------------

lhprove_bind_assumptions([],Env,Env,P,P).
lhprove_bind_assumptions([A|As],Env1,Env3,l(X,P),Body):-
  lhprove_bind_assumptions(As,[X:A|Env1],Env3,P,Body).

%% proof-producing premise handler (same name as existing hprove_imp/3)
%% premise is a nested Horn clause: prove it (yields a lambda term)
hprove_imp((D:-Cs),_B,Vs,Proof):- !, lhprove_env((D:-Cs),Vs,Proof).
%% premise is atomic/var: retrieve its witness from the environment
hprove_imp(A,_B,Vs,Proof):- memberchk(Proof:A,Vs).

/*
Example (same as in hprove/1, now with proof term):

?- H=(r:-[(r:-[p, q]), (q:-[p]), p]), lhprove(H,P).
P = l(_R, l(_Q, l(_P, a(a(_R, _P), a(_Q, _P))))).

This corresponds to: \R.\Q.\P. R P (Q P)
and mirrors the curried shape of the Herbrand-style application order.
*/
