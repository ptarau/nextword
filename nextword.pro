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

%% converts list to implication chain
% also handling special cases of empty and single-element lists
list2impl([],[]).
list2impl([X],X).
list2impl([X,Y|Xs],R):-seq2left_heavy([Y|Xs],X,R).

%% converts sequence to left-heavy implication chain
%% e.g. [a,b,c] -> (a->b)->c
seq2left_heavy([],End,End).
seq2left_heavy([X|Xs],Chain,End):-seq2left_heavy(Xs,(Chain->X),End).

%% extracts prefix of implication chain
ipref(X,X).
ipref(Xs->_,Ys):-
  ipref(Xs,Ys).

%% extracts suffix of implication chain
isuff(X,Y):-isuff0(X,Y).
isuff(X,X).

isuff0(_->X,X).
isuff0(Xs->X,(Ys->X)):-
  isuff0(Xs,Ys).

%% extracts a prefix of a suffix of implication chain
isufpref-->isuff,ipref.

%% stores implication chains of given sentences in the database
:-dynamic isent/1.

store_impls(Sents):-
  retractall(isent(_)),
  member(Sent,Sents),
  distinct(Impl,sent2impl(Sent,Impl)),
  assertz(isent(Impl)),
  fail;true.

%% converts sentence (seen as a long prolog atom) to implication chain
%% made of words (also prolog atoms)
sent2impl(Sent,R):-
  atomic_list_concat(Words,' ',Sent),
  list2impl(Words,R).

%% querying with text string a gainst chains database
qa(TextQuery,ISent):-
    string_lower(TextQuery,Query1),
    atom_string(QueryAtom,Query1),
    atomic_list_concat(Words,' ',QueryAtom),
    lqa(Words,ISent).

%% querying with list of words against chains database
lqa(Words,ISent):-
    list2impl(Words,LeftImplQuery),
    iqa(LeftImplQuery,ISent).

 %% finds in database a sentence whose implication chain
 %% has as prefix of a suffix ainside an implication chain in the dbase
iqa(LeftImplQuery,ISent):-
    isent(ISent), % assumes isent/1 database
    distinct(ISent,isufpref(ISent,LeftImplQuery)).

%% top-level querying with text string
qa(Query):-
    qa(Query,ISent),
    % writeq(ISent),nl,
    impl2list(ISent,Words),
    atomic_list_concat(Words,' ',Answer),
    writeq(Answer),nl,nl,
    fail;true.

%% converts implication chain to list
impl2list(E,Xs):-impl2list(E,Xs,[]).
    
impl2list(E->X)-->!,impl2list(E),[X].
impl2list(X)-->[X].

store_impls :-
  Sents=['the cat sits on the mat',
    'the dog sits on the log',
    'the cat chases the mouse',
    'the dog chases the cat'],
  store_impls(Sents),
  listing(isent/1).

/*
% a few tests

?- iprove(((p->q)->r) -> (p->q->r)).
true.

?- lprove(((p->q->r)->(p->q)->p->r),SCombinator).
SCombinator = l(_A, l(_B, l(_C, a(a(_A, _C), a(_B, _C)))))

?- lprove(((p->q)->r) -> (p->q->r), LambdaExpr).
LambdaExpr = l(_A, l(_, l(_B, a(_A, l(_, l(_, _B)))))).

?- sent2impl('the cat sits on the mat',Impl).
Impl = (((((the->cat)->sits)->on)->the)->mat).



?- store_impls.
:- dynamic isent/1.

isent((((((the->cat)->sits)->on)->the)->mat)).
isent((((((the->dog)->sits)->on)->the)->log)).
isent(((((the->cat)->chases)->the)->mouse)).
isent(((((the->dog)->chases)->the)->cat)).


?- qa('the cat').
'the cat sits on the mat'

'the cat chases the mouse'

'the dog chases the cat'

?- lqa([the,X,chases],R).
X = cat,
R = ((((the->cat)->chases)->the)->mouse) ;
X = dog,
R = ((((the->dog)->chases)->the)->cat) .

*/    

