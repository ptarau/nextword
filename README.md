# Next word prediction as logical inference

The file *arrow.pl* implements a Prolog specification of the *arrow.py* neural net trainer and inferencer.

Both show the retrieval of the context and the actual sentence matvching the query.

As our tokens are lower case words, they need to be an exact subsequence occurring in
one of the sentences for sucessful retrival.

to fetch files from guttenberg.org to ```data``` folder do:

```
python guttenberg.py
```

The try out for the Prolog-based QA:

```bash
swipl -s arrow.pl

?- query(war_and_peace,napoleon).
...
?-query(dracula,'the castle').
...

```

After training with ```arrow.py``` on the files in ```data```:

```bash
./train.sh the_ayes
...
./train.sh dracula
...
./train.sh war_and_peace
...
```

and for inference with ```arrow.py``` on the trained checkpoints:

```bash
./generate.sh the_ayes
...
./generate.sh dracula
...
./generate.sh war_and_peace
...
```

#### Enjoy,

Paul Tarau

Jan 1, 2026

