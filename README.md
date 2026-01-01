# Next word prediction as logical inference

The file *arrow.pl* implements a Prolog specification of the *arrow.py* neural net trainer and inferencer.

Both show the retrieval of the context and the actual sentence matvching the query.

As our tokens are lower case words, they need to be an exact subsequence occurring in
one of the sentences for sucessful retrival.

to fetch files from guttenberg.org
use guttenberg.py

Try out:

```bash
swipl -s arrow.pl

go1.
...
go5.

```

Try

```bash
t_guermantes.sh

t_war_and_peace.sh

t_eyes.sh*
```

for training with ```arrow.py```

and

```bash
i_guermantes.sh

i_war_and_peace.sh

i_arrow.sh
```

for inference with ```arrow.py```.
