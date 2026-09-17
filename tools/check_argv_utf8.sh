#!/bin/sh
# Accents on the command line reach the engine as UTF-8 (docs/LEZIONI.md #13).
# The text lives in this file, not in the Makefile: GNU make on Windows hands its recipes to the
# shell in the local code page and would mangle it before the engine sees it (#36).
# The expected ids are transformers' for this text with the OLMoE tokenizer.
#   tools/check_argv_utf8.sh <trochilus binary> <vocab.gguf>
bin="$1"
vocab="$2"
ids=$("$bin" tokenize -m "$vocab" -p "città è perché" 2>&1)
rc=$?
if [ $rc -eq 126 ]; then
    echo "== argv UTF-8: SKIPPED, Smart App Control blocked the new exe (docs/LEZIONI.md #12)"
elif [ "$ids" = "68,770,5991,12187,591,32716" ]; then
    echo "== argv UTF-8 ok"
else
    echo "argv UTF-8 broken (docs/LEZIONI.md #13): got '$ids'"
    exit 1
fi
