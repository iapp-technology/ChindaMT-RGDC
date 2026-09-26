# Source corpora

The RGDC pool is built from the ten public English-Thai parallel corpora below, each used in both
directions (en2th and th2en), about 17.85M records in total. Counts are records per direction in
the pool the paper used; use them to confirm you obtained the same version of each corpus.

| Corpus | Records per direction | Citation (see the paper's references) | Where to obtain |
|---|---:|---|---|
| hplt | 4,088,354 | de Gibert et al., 2024 | HPLT parallel data release, en-th (hplt-project.org / OPUS `HPLT`) |
| paracrawl | 2,175,889 | Koehn, 2024 | Neural ParaCrawl en-th (paracrawl.eu) |
| xlent | 1,236,145 | El-Kishky et al., 2021 | OPUS `XLEnt`, en-th |
| scb_2020 | 1,001,752 | Lowphansirikul et al., 2020 | SCB-MT-EN-TH-2020, twelve sub-corpora (AIResearch release) |
| opensubtitles | 206,563 | Lison and Tiedemann, 2016 | OPUS `OpenSubtitles`, en-th |
| bible | 124,386 | Christodoulopoulos and Steedman, 2015 | Multilingual Bible corpus, en-th |
| en-th-texts | 59,859 | kvush, en-th texts | Hugging Face dataset `kvush/eng_th_texts` (file `eng_th_texts.csv`) |
| wikimedia | 32,999 | Tiedemann, 2012 | OPUS `wikimedia`, en-th |
| tatoeba | 1,184 | Tiedemann, 2020 | Tatoeba, en-th sentence pairs |
| elrc | 237 | Tiedemann, 2012 | OPUS `ELRC` en-th collections |

Each corpus was exported to a two-column file of English and Thai segments and converted with
`src/data/prepare_corpus.py` into the minimal ShareGPT layout, then with
`src/data/generate_mid_zero_shot.py` into the instruction-diversified `mid_zero-shot` variant that
Phase 1 consumes. The converters reproduce the pool files byte for byte from the same pairs, with one known
exception documented in `src/data/prepare_corpus.py` (a single wikimedia row with a missing cell).
