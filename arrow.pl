% :-include('create_sents.pl').
:-include('impl.pl').

%% converts sentence (seen as a long prolog atom) to implication chain
%% made of words (also prolog atoms)
sent2impl(Sent,R):-
   atomic_list_concat(Words,' ',Sent),
   list2impl(Words,R).

% converts list to implication chain
% also handling special cases of empty and single-element lists
list2impl([],[]).
list2impl([X],X).
list2impl([X,Y|Xs],R):-seq2left_heavy([Y|Xs],X,R).

%% converts sequence to left-heavy implication chain
%% e.g. [a,b,c] -> a->(b->c)
seq2left_heavy([],End,End).
seq2left_heavy([X|Xs],Chain,End):-seq2left_heavy(Xs,(Chain->X),End).

%% converts implication chain to list
impl2list(E,Xs):-impl2list(E,Xs,[]).

impl2list(E->X)-->!,impl2list(E),[X].
impl2list(X)-->[X].

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

%% extracts a prefix of a suffixof implication chain
isufpref-->isuff,ipref.


infer_from(QF,Formula,Pref+Suff) :-
    isuff(Formula,Suff),QF\==Suff,
    ipref(Suff,Pref),QF\==Pref,QF\==Suff,
    iprove(QF,[Pref]).


infer_from_pref(QF,Formula,Pref+QF) :-
    ipref(Formula,Pref),QF\==Pref,
    iprove(Pref->QF).

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

:-dynamic(isent/1).

%% stores implication chains of given sentences in the database
store_impls :-
   Sents=['the cat sits on the mat',
          'the dog sits on the log',
          'the cat chases the mouse',
          'the dog chases the cat'],
   store_impls(Sents),
   listing(isent/1).

%% stores implication chains of given sentences in the database
store_impls(Sents):-
   retractall(isent(_)),
   member(Sent,Sents),
   distinct(Impl,sent2impl(Sent,Impl)),
   assertz(isent(Impl)),
   fail;true.

%% queries implication chains database using implication chain
query_impls(Q):-
   store_impls,
   writeq(Q),nl,
   qa(Q).

%% example query
query_impls:-
  Q='the cat sits',
  query_impls(Q).

%% example of query with logic variables
 query_with_vars:-
   iqa(((_->sits)->on),R),
    writeq(R),nl.


%% stores implication chains of given sentences in the database
to_impl_db(InputFile):-
   retractall(isent(_)),
   file2sents(InputFile,Sent),
   string_lower(Sent,SentLower),
   sent2impl(SentLower,Impl),
   assertz(isent(Impl)),
   fail;true.

%% file2sents(+OutputFile, -SentAtom) is nondet.
%  Backtracks over lines in OutputFile, yielding each as an atom.
file2sents(OutputFile, Sent) :-
    setup_call_cleanup(
        open(OutputFile, read, In, [encoding(utf8)]),
        line_atom(In, Sent),
        close(In)
    ).

% --- backtracking line reader over a stream ---
line_atom(In, Atom) :-
    repeat,
      read_line_to_string(In, S),
      ( S == end_of_file
      -> !, fail
      ; string_to_atom(S, Atom)
      ).



qa_repl(QA) :-
    format("Type a sentence, then press Enter. Empty line or EOF quits.~n", []),
    qa_loop(QA).

qa_loop(QA) :-
    format("> ", []),
    read_line_to_string(user_input, S),
    ( S == end_of_file
    -> true
    ; S == ""
    -> true
    ; call_qa(QA, S),
      qa_loop(QA)
    ).

% Calls qa/1 with an atom (or change to qa/1 expecting string if you prefer)
call_qa(QA,S) :-
    string_to_atom(S, A),
    catch(
        call(QA,A),
        E,
        ( print_message(error, E),
          fail
        )
    ).


go1:-
  %file2sents_to_file('data/kafka.txt','data/kafka_sents.txt'),
  to_impl_db('data/kafka_sents.txt'),
  qa('like a dog'),
  qa_repl(qa).

go2:-
  %file2sents_to_file('data/war_and_peace.txt','data/war_and_peace_sents.txt'),
  to_impl_db('data/war_and_peace_sents.txt'),
  qa('why did Kutuzov'),
  qa_repl(qa).

go3:-
  %file2sents_to_file('data/guermantes.txt','data/guermantes_sents.txt'),
  to_impl_db('data/guermantes_sents.txt'),
  qa('princess'),
  qa_repl(qa).

go4:-
  %file2sents_to_file('data/guermantes.txt','data/guermantes_sents.txt'),
  to_impl_db('data/crystal_sents.txt'),
  qa('the pilot'),
  qa_repl(qa).

go5:-
  %file2sents_to_file('data/guermantes.txt','data/guermantes_sents.txt'),
  to_impl_db('data/the_eyes_sents.txt'),
  qa('his eyes'),
  qa_repl(qa).
