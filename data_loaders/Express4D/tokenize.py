# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import spacy
nlp = spacy.load('en_core_web_sm')
# Tokenizer according to https://github.com/EricGuo5513/HumanML3D/blob/main/text_process.py
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    
    tokens = " ".join(f"{x}/{y}" for x, y in zip(word_list, pos_list))
    output = sentence + '#' + tokens
    return output

if __name__ == '__main__':
    with open('dataset/Express4D/texts_plain.txt', "r") as f:
        lines = f.readlines()
        new_lines = [process_text(txt.replace('\n','')) for txt in lines]
    with open('dataset/Express4D/texts.txt', "w") as f:
        f.write("\n".join(new_lines))

