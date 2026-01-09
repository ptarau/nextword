

c:-make.

i2t-->to_horn,horn2term. % assumes vars only as args

t2i-->term2horn,from_horn.

to_horn(H,H):-var(H),!.
to_horn(H,H):-atomic(H),!.
to_horn((A->B),(H:-Bs)):-!,to_horns((A->B),Bs,H).

to_horns(H,[],H):-var(H),!.
to_horns(H,[],H):-atomic(H),!.
to_horns((A->B),[HA|Bs],H):-to_horn(A,HA),to_horns(B,Bs,H).

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