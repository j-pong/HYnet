import editdistance

def calculate_accuracy(y_hat, y_true, word_eds, word_ref_lens, char_eds, char_ref_lens):
    # calculate sentence distance between hyp and ref
    seq_hat_text = y_hat
    seq_true_text = y_true

    hyp_words = seq_hat_text.split()
    ref_words = seq_true_text.split()
    word_eds.append(editdistance.eval(hyp_words, ref_words))
    word_ref_lens.append(len(ref_words))

    hyp_chars = seq_hat_text.replace(' ', '')
    ref_chars = seq_true_text.replace(' ', '')
    char_eds.append(editdistance.eval(hyp_chars, ref_chars))
    char_ref_lens.append(len(ref_chars))

    return word_eds, word_ref_lens, char_eds, char_ref_lens

word_eds = []
word_ref_lens = []
char_eds = []
char_ref_lens = []

with open('hyp.txt', 'r', encoding="utf-8") as f:
    with open('ref.txt', 'r', encoding="utf-8") as g:
        hyps = f.readlines()
        refs = g.readlines()

        for i in range(len(hyps)):
            y_hat = hyps[i]
            y_true = refs[i]
            word_eds, word_ref_lens, char_eds, char_ref_lens = calculate_accuracy(y_hat, y_true, word_eds, word_ref_lens, char_eds, char_ref_lens)

        cer = float(sum(char_eds))*100 / sum(char_ref_lens)
        wer = float(sum(word_eds))*100 / sum(word_ref_lens)

with open('CER_WER.txt', 'w') as f:
    f.write(format(wer, '.2f') + "/" + format(cer, '.2f'))
