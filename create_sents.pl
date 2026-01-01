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


