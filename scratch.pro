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

tprove(A):-tprove(A,[]).

tprove(A,Vs):-memberchk(A,Vs),!.
tprove((BAs),Vs1):-BAs=..[B|As],!,append(As,Vs1,Vs2),tprove(B,Vs2).
tprove(G,Vs1):-          % atomic(G), G not on Vs1
  member((GXs),Vs1),functor(GXs,G,_),!, % if not, we just fail
  select((BAs),Vs1,Vs2), % outer select loop
  BAs=..[B|As],
  select(A,As,Bs),         % inner select loop
  tprove_imp(A,B,Vs2), % A element of the body of B
  !,
  ttrimmed((B:-Bs),NewB), % trim off empty bodies
  tprove(G,[NewB|Vs2]).

tprove_imp(DCs,B,Vs):- functor(DCs,D,_),!,BD=..[B,D],tprove(DCs,[BD|Vs]).
tprove_imp(A,_B,Vs):-memberchk(A,Vs).

ttrimmed((B:-[]),R):-!,R=B.
ttrimmed(BBs,BBs).

selarg(X,FXs,FYs):-
   functor(FXs,F,_N),
   arg(I,FXs,X),
   findall(Y,(arg(J,FXs,Y),I=\=J),Ys),
   FYs=..[F|Ys].
   
membarg(X,FXs):-arg(_I,FXs,X).

membfun(F,FXs):-functor(F,FXs,_).

argpend(FXs,FYs,FZs):-
   functor(FXs,F,_),
   findall(X,(arg(_,FXs,X);arg(_,FYs,X)),Zs),
   FZs=..[F|Zs].
   
   