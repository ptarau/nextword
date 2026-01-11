

% c:-make.

%% converts an implication chain to a Herbrand term
implicational2herbrand-->implicational_to_horn,horn2term. % assumes vars only as args

%% converts a Herbrand term to an implication chain
herbrand2implicational-->term2horn,horn_to_implicational.

%% converts implication chain like a->b->f to f:-[a,b]
implicational_to_horn(H,H):-var(H),!.
implicational_to_horn(H,H):-atomic(H),!.
implicational_to_horn((A->B),(H:-Bs)):-!,to_horns((A->B),Bs,H).

% helper predicate for implicational_to_horn
to_horns(H,[],H):-var(H),!.
to_horns(H,[],H):-atomic(H),!.
to_horns((A->B),[HA|Bs],H):-implicational_to_horn(A,HA),to_horns(B,Bs,H).

%% converts horn clause like f:-[a,b] to a->b->f
horn_to_implicational(A,B):-implicational_to_horn(B,A).

%% converts a  Horn like f(a,b) to a Horn notation like f:-[a,b]
term2horn(T,H):-var(T),!,H=T.
term2horn(T,H):-atomic(T),!,H=T.
term2horn(T,(F:-Ys)):-T=..[F|Xs],
  maplist(term2horn,Xs,Ys).

%% converts a  Horn like f:-[a,b] to f(a,b)
horn2term(H,T):-var(H),!,T=H.
horn2term(H,T):-atomic(H),!,T=H.
horn2term((F:-Ys),T):-
  maplist(horn2term,Ys,Xs),
  T=..[F|Xs].

/*

?- implicational2herbrand(a->b->f,Herb).
Herb = f(a, b).

?- implicational2herbrand((x->g)->f,Herb).
Herb = f(g(x)).

?- implicational_to_horn((x->g)->f,Herb).
Herb = (f:-[(g:-[x])]).

?- implicational_to_horn((x->g)->f,Herb).
Herb = (f:-[(g:-[x])]).

*/