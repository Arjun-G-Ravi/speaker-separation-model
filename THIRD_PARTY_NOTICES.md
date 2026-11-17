# Dataset and method attribution

Dataset sources are fetched only when the user runs preparation. No speech is bundled.

- LibriSpeech: Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur,
  “LibriSpeech: An ASR corpus based on public domain audio books,” ICASSP 2015.
  Source: https://www.openslr.org/12; audio license: CC BY 4.0.
- LibriMix: Joris Cosentino, Manuel Pariente, Samuele Cornell, Antoine Deleforge,
  Emmanuel Vincent, “LibriMix: An Open-Source Dataset for Generalizable Speech
  Separation,” 2020. https://arxiv.org/abs/2005.11262.
  Metadata and generation recipe: https://github.com/JorisCos/LibriMix.
- Conv-TasNet: Yi Luo and Nima Mesgarani, “Conv-TasNet: Surpassing Ideal
  Time-Frequency Magnitude Masking for Speech Separation,” IEEE/ACM TASLP 2019.
  https://arxiv.org/abs/1809.07454. Model code here is implemented from scratch.
- A-FRCNN: Xiaolin Hu, Kai Li, Weiyi Zhang, Yi Luo, Jean-Marie Lemercier,
  Timo Gerkmann, “Speech Separation Using an Asynchronous Fully Recurrent
  Convolutional Neural Network,” NeurIPS 2021.
  https://papers.nips.cc/paper/2021/file/be1bc7997695495f756312886f566110-Paper.pdf.
  `src/afrcnn.py` adapts the Method B / concatenation reference implementation:
  https://github.com/JusperLee/AFRCNN-For-Speech-Separation/blob/master/AFRCNN.py.
  Normalization reductions use float32 here, and inputs use the project's `[B,T]`
  interface. Upstream license follows.
- Earlier speaker split: `src/earlier_speaker_split/model.py` reuses the A-FRCNN
  adaptation above and adds an independently implemented convolutional speaker
  split and shared reconstruction. The design is inspired by Ui-Hyeop Shin,
  Sangyoun Lee, Taehan Kim, and Hyung-Min Park, “Separate and Reconstruct:
  Asymmetric Encoder-Decoder for Speech Separation,” NeurIPS 2024.
  https://arxiv.org/abs/2406.05983. No SepReformer source code is copied.

## Upstream A-FRCNN license

MIT License

Copyright (c) 2021 Kai Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Upstream LibriMix license

MIT License

Copyright (c) 2020 JorisCos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
