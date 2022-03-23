# CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks

Official implementation of CLEval | [paper](https://arxiv.org/abs/2006.06244)


## 사용방법

### Requirements
* python 3.x
* see requirements.txt file to check package dependency. To install, command
```
pip3 install -r requirements.txt
```

### XML파일 사용시
```
python script.py  -g= [gt파일명].xml -s=[result파일명].xml --E2E --BOX_TYPE=LTRB
```

### 기존방법으로 txt파일 사용시 : .zip 파일이어야함
```
python script.py  -g= [gt파일명].zip -s=[result파일명].zip --E2E --BOX_TYPE=LTRB --XML=False
```


### Paramter list
<!-- 
### Paramters for evaluation script
| name | type | default | description |
| ---- | ---- | ------- | ---- |
| -g | ```string``` | | path to ground truth zip file |
| -s | ```string``` | | path to result zip file |
| -o | ```string``` | | path to save per-sample result file 'results.zip' | -->

| name | type | default | description |
| ---- | ---- | ------- | ---- |
| --BOX_TYPE | ```string``` | ```QUAD``` | annotation type of box (LTRB, QUAD, POLY) |
| --TRANSCRIPTION | ```boolean``` | ```False``` | set True if result file has transcription |
| --CONFIDENCES | ```boolean``` | ```False``` | set True if result file has confidence |
| --E2E | ```boolean``` | ```False``` | to measure end-to-end evaluation (if not, detection evalution only) |
| --CASE_SENSITIVE | ```boolean``` | ```True``` | set True to evaluate case-sensitively. (only used in end-to-end evaluation) |
| --XML| ```boolean``` | ```True``` | set file extension |

* Note : Please refer to ```arg_parser.py``` file for additional parameters and default settings used internally.

## License

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
