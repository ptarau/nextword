retrive(Key,Value):-
   Key=[_|_],
   append(Prefix,_,Value),
   append(_,Key,Prefix).
   
%% extracts prefix of implication chain
ipref(X,X).
ipref((Xs->_),Ys):-
  ipref(Xs,Ys).

%% extracts suffix of implication chain

isuf(X,Y):-isuf_(X,Y).
isuf(X,X).

isuf_(_->X,X).
isuf_((Xs->X),(Ys->X)):-
  isuf_(Xs,Ys).

%% extracts a prefix of a suffix of implication chain
isufpref-->isuf,ipref.
isp(Value,Key):-isp(Value,Suf),isp(Suf,Key).

iretrieve(Key,Value):-
   isuf(Value,SubForm),
   ipref(SubForm,Key).
   
   