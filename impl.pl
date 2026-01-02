%% simplest, with multisets, no contraction
%% with a single select/3 operation
:-include(stats).

iprove(T):-iprove(T,[]).

iprove(A,Vs):-memberchk(A,Vs),!.
iprove((A->B),Vs):-!,iprove(B,[A|Vs]).
iprove(G,Vs1):-
  select((A->B),Vs1,Vs2),
  iprove_imp(A,B,Vs2),
  !,
  iprove(G,[B|Vs2]).

iprove_imp((C->D),B,Vs):-!,iprove((C->D),[(D->B)|Vs]).
iprove_imp(A,_,Vs):-memberchk(A,Vs).

%% Horn term representation of chained implications

hprove_(Impl):-to_horn(Impl,Horn),hprove(Horn).

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


%% reversible translator between implicational
%% end embedded Horn clause form

to_horn((A->B),(H:-Bs)):-!,to_horns((A->B),Bs,H).
to_horn(H,H).

to_horns((A->B),[HA|Bs],H):-!,to_horn(A,HA),to_horns(B,Bs,H).
to_horns(H,[],H).

% converters between f(...) and f:-[..]
to_term-->to_horn,horn2term.

from_term-->term2horn,from_horn.


from_horn(A,B):-to_horn(B,A).

term2horn(T,H):-var(T),!,H=T.
term2horn(T,H):-atomic(T),!,H=T.
term2horn(T,(F:-Ys)):-T=..[F|Xs],
  maplist(term2horn,Xs,Ys).

horn2term(H,T):-var(H),!,T=H.
horn2term(H,T):-atomic(H),!,T=H.
horn2term((F:-Ys),T):-
  maplist(horn2term,Ys,Xs),
  T=..[F|Xs].

tprove(T):-
  term2horn(T,(H:-Bs)),
  hprove(H,Bs).



%% variant of iprove that produces
%% lambda terms as proofs

lprove(T):-lprove(T,_).

%% X is the lambda term proof of T
%% or, via Curry-Howard, X:T as X is an inhabitant of type T
lprove(T,X):-lprove(X,T,[]),!.

lprove(X,A,Vs):-memberchk(X:A,Vs),!. % leaf variable

lprove(l(X,E),(A->B),Vs):-!,lprove(E,B,[X:A|Vs]).  % lambda term

lprove(E,G,Vs1):-
  % member(_:V,Vs1),head_of(V,G),!, % fail if non-tautology
  select(S:(A->B),Vs1,Vs2),       % source of application
  lprove_imp(T,A,B,Vs2),          % target of application
  !,
  lprove(E,G,[a(S,T):B|Vs2]).     % application

lprove_imp(l(X,E),(C->D),B,Vs):-!,lprove(E,(C->D),[X:(D->B)|Vs]).
lprove_imp(E,A,_,Vs):-memberchk(E:A,Vs).

head_of(_->B,G):-!,head_of(B,G).
head_of(G,G).




show_proof(A,B):-
  %%qqq(B),
  write('\\verb|'),write(A->B),write('|'),nl,
  qqq(A->B),
  lprove((A->B),Proof),
  to_lambda(Proof),
  nl.
  /*
  to_term(A,AT),
  to_term(B,BT),
  to_term(A->B,ABT),
  ppp(AT),nl,
  ppp(BT),nl,
  ppp(ABT),nl,
  qqq(ABT),nl
  */


% tests

% implication as Horn term
it1:-
  Impl=((p->q) -> ((p->q)->r) ->r),
  to_horn(Impl,Horn),
  horn2term(Horn,Term),
  write(Impl),nl,
  write(Term),nl,
  tprove(Term).

% left chain as Prolog term - no lists involved!
it2:-
  Impl=(((((the->cat)->sits)->on)->the)->mat),
  to_term(Impl,Term),
  write(Impl),nl,
  ppt(Term),nl.

% seq implies bloc to bloc implication
sh3:-
  ppp('Sequence implies Block to Block'),nl,
  X=(((((p->q)->r)->a)->b)->c),
  Y=((p->q)->r),
  Z=((a->b)->c),
  show_proof(X,(Y->Z)).

it3:-
  X=(((((p->q)->r)->a)->b)->c),
  Y=((p->q)->r),
  Z=((a->b)->c),
  Impl=(X->(Y->Z)),
  lprove(Impl,Proof),
  portray_clause((Proof:Impl)),nl,
  to_term(Impl,IT),
  write(it),nl,
  ppt(IT),
  to_term(X,XT),
    write(xt),nl,
  to_term(Y,YT),
    write(yt),nl,
  to_term(Z,ZT),
    write(zt),nl,
  ppt(XT),
  ppt(YT),
  ppt(ZT).

it3a:-
  X=(((('E'->a)->b)->c)),
  Z=((a->b)->c),
  Impl=('E'->(X->Z)),
  lprove(Impl,Proof ),
  qqq(Proof),
  to_lambda(Proof).


it3b:-
  X=(((('E'->p)->q))),
  Y=(p->q),
  Impl=(X->Y),
  lprove(Impl,Proof ),
  qqq(Proof),
  ppt(Impl),
  ppt(Proof),
  to_lambda(Proof).



% modus ponens
sh4:-
  ppp('Modus Ponens'),nl,
  show_proof(p,(p->q)->q).

it4:-
  T=((p->(p->q)->q)),
  lprove(T,Proof),
  portray_clause((Proof:T)),nl.


sh5:-
  ppp('K combinator'),nl,
  show_proof(p,(q->p)).

% K
it5:-
  T=(p->(q->p)),
  lprove(T,Proof),
  portray_clause((Proof:T)),nl,
  ppt(T),nl,
  to_lambda(Proof).


sh6:-
  ppp('S combinator'),nl,
  A=(p->(q->r)),
  B=((p->q)->(p->r)),
  show_proof(A,B),
  show_proof(B,A).


sh6a:-
  ppp('S combinator'),nl,
  A=(db->(query->answer)),
  B=((db->query)->(db->answer)),
  show_proof(A,B),
  show_proof(B,A).


% S
% l(A, l(B, l(C, a(A, l(_, l(D, a(C, a(D, l(_, B)))))))))
it6:-
  T=((p->q->r)->(p->q)->(p->r)),
  lprove(T,Proof),
  portray_clause((Proof:T)),nl,
    ppt(T),
     to_lambda(Proof).


shlr:-
    ppp('Left nested implies right nested'),nl,
    Left=((((a->b)->c)->d)->e),
    Right=(a->(b->(c->(d->e)))),
    show_proof(Left,Right).

left_impl_right:-
  Left=((((a->b)->c)->d)->e),
  Right=(a->(b->(c->(d->e)))),
  Impl=(Left->Right),
  lprove(Impl,Proof),
  ppl(Proof),
  ppt(Impl),
  to_term(Impl,IT),
  write(it),nl,
  ppt(IT).


ppl(X):-
  ppt(X),nl,
  to_lambda(X).



% formula used by iprove to reduce left-nested (p->q)->r
vorobiev:-
   % (A->B)->A
   X=(((c -> d) -> b) -> (c->d)),
   Y=((d->b) -> (c->d)),
   T=(X->Y),
   TT=(Y->X),
   lprove(T,R),
   portray_clause((T:-R)),nl,
   lprove(TT,RR),
   portray_clause((TT:-RR)),nl,
   to_term(X,XT),
   to_term(Y,YT),
   to_term((X->Y),XY),
   to_term((Y->X),YX),
    write(xt),nl,
    ppt(XT),
    write(yt),nl,
    ppt(YT),
    ppt(XT=YT),
    ppt(XY),
    ppt(YX),
    ppt(TT).

vorob4:-
   X=(c-> ((c -> d) -> b) ->d),
   Y=(c-> (d->b) ->d),
   T=(Y->X),
   iprove(T).

vorob5:-
   X=(c-> ((c -> d) -> b)),
   Y=(c-> (d->b)),
   T=(Y->X),
   iprove(T).

vorob6:-
  X=(c -> (c->d)),
  Y=(c -> d),
  T=(Y->X),
  iprove(T).

nested_implies_simple:-
   A=(('E'->q)->r),B=(q->r),lprove((A->B),Proof),
   ppt(A->B),
   ppt(Proof),nl,
   qqq(Proof),
   to_lambda(Proof).

other_impl:-
  A=(p->q),
  B=((q->r)->(p->r)),
  lprove((A->B),Proof),
  ppt(A->B),
  ppt(Proof),nl,
  qqq(Proof).

c:-make.
