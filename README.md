BioDataLab
readme please

Background execute : nohup python3 -u DNN_cancer_conf.py > ./logfile_name & 
option -u : logging for print()  

<Data Step>

MergeToyFile:
해당 path를 인풋으로 받고에 폴더에 있는 모든 파일을 긁어와 cancer 여부와 gene 발현량 데이터를 리스트로 묶어 아웃풋으로 내보낸다. 

MergeUntil:
MergeToyFile의 sub-function이다.

1. 이 두 함수로 데이터를 불러온다.
2. 데이터에 중복된 유전자와 이름이 없는 유전자 데이터를 제거한다.
3. 로그스케일을 한다: 
데이터 중 1이하의 값을 가지는 경우는 로그를 씌울 때 곤란해서 다 1로 만들어 주고 로그 스케일이 된 데이터와 그렇지 않은 데이터가 있으므로 한 사람의 유전자 발현량의 max값이 100을 넘을 경우 로그 스케일이 되지 않았다고 가정하고 해당 GSM patients의 유전자 발현량에 대해서 로그2 스케일을 해준다.
4. Z-scale Normalization을 해준다.
5. 데이터의 크기를 줄이기 위해 소숫점 3자리까지만 살린다.
6. 유전자의 분산을 구해 Toy1000$VAR에 저장한다.
7. 이 중 분산이 큰 상위 6000개의 유전자만 살릱다.
8. 데이터를 모델에 넣을 수 있는 형태로 transpose한다.
9. 이때 유전자의 이름을 나타내주는 것은 Gene Symbol만 살린다.
10. five-fold 인덱스와  cancer 여부를 추가한다.
10. 완성된 데이터 프레임은 다음과 같은 구조를 가진다.
행: GSM
열: 유전자 6000개, index, result(암 여부)
11. 마지막으로 GSM 넘버가 없는 데이터 들을 제거 해준다.

