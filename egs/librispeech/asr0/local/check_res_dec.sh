#!/bin/bash

# 2018 Mirco Ravanelli Univeristé de Montréal - Mila
# 2020 Modified KGGEJHC

for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | local/best_wer.sh; done
for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/*score_*/*.sys 2>/dev/null | local/best_wer.sh; done
exit 0

# TODO: Maybe in this way?
# for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* | local/best_wer.sh 2>/dev/null; done
# for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/*score_*/*.sys | local/best_wer.sh 2>/dev/null; done