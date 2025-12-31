:- use_module(library(readutil)).  % read_line_to_string/2

%% file2sents_to_file(+InputFile, +OutputFile) is det.
%  Calls: python3 file2sents_cli.py InputFile OutputFile
file2sents_to_file(Input, Output) :-
    shell_quote(Input,  QIn),
    shell_quote(Output, QOut),
    format(string(Cmd), 'python3 to_sent_file.py ~w ~w', [QIn, QOut]),
    shell(Cmd, Exit),
    ( Exit =:= 0
    -> true
    ;  throw(error(python_file2sents_failed(Exit, Cmd), _))
    ).

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

% --- minimal safe quoting for shell args ---
shell_quote(S, Quoted) :-
    ( atom(S) -> atom_string(S, Str) ; Str = S ),
    escape_for_dq(Str, Esc),
    format(string(Quoted), '"~s"', [Esc]).

escape_for_dq(Str, Esc) :-
    string_codes(Str, Cs),
    phrase(esc_dq(Cs), Out),
    string_codes(Esc, Out).

esc_dq([]) --> [].
esc_dq([0'\\|T]) --> [0'\\,0'\\], esc_dq(T).
esc_dq([0'" |T]) --> [0'\\,0'"],  esc_dq(T).
esc_dq([C   |T]) --> [C],         esc_dq(T).



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