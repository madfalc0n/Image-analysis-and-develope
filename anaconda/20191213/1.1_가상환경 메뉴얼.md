# 아나콘다 가상환경 설치

### 0. 기타 명령어
 - conda --version 
	- 아나콘다 버전 확인
 - conda search python
	- 설치된 파이선 버전 정보 확인
- conda list
  - 설치된 패키지 정보 불러오기

### 1. 설치
 - conda create -n [이름] python=[버전]
	- conda create -n python374 python=3.7.4
```bash
(base) C:\Users\student>conda create -n python374 python=3.7.4
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.7.12
  latest version: 4.8.0

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: C:\Users\student\Anaconda3\envs\python374

  added / updated specs:
    - python=3.7.4


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    ca-certificates-2019.11.27 |                0         163 KB
    certifi-2019.11.28         |           py37_0         157 KB
    vs2015_runtime-14.16.27012 |       hf0eaf9b_1         1.1 MB
    ------------------------------------------------------------
                                           Total:         1.4 MB

The following NEW packages will be INSTALLED:

  ca-certificates    pkgs/main/win-64::ca-certificates-2019.11.27-0
  certifi            pkgs/main/win-64::certifi-2019.11.28-py37_0
  openssl            pkgs/main/win-64::openssl-1.1.1d-he774522_3
  pip                pkgs/main/win-64::pip-19.3.1-py37_0
  python             pkgs/main/win-64::python-3.7.4-h5263a28_0
  setuptools         pkgs/main/win-64::setuptools-42.0.2-py37_0
  sqlite             pkgs/main/win-64::sqlite-3.30.1-he774522_0
  vc                 pkgs/main/win-64::vc-14.1-h0510ff6_4
  vs2015_runtime     pkgs/main/win-64::vs2015_runtime-14.16.27012-hf0eaf9b_1
  wheel              pkgs/main/win-64::wheel-0.33.6-py37_0
  wincertstore       pkgs/main/win-64::wincertstore-0.2-py37_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
vs2015_runtime-14.16 | 1.1 MB    | ############################################################################ | 100%
ca-certificates-2019 | 163 KB    | ############################################################################ | 100%
certifi-2019.11.28   | 157 KB    | ############################################################################ | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate python374
#
# To deactivate an active environment, use
#
#     $ conda deactivate


(base) C:\Users\student>
```

### 2. 제거
 - conda remove -name [이름] --all
```bash
(base) C:\Users\student>conda remove -name python374 --all

(base) C:\Users\student>
```

### 3. 설치확인
 - conda info --envs ,모든 가상환경 리스트 출력
```bash
 (base) C:\Users\student>conda info --envs
# conda environments:
#
base                  *  C:\Users\student\Anaconda3
python374                C:\Users\student\Anaconda3\envs\python374
```

 - conda info ,activate 되어있는 가상환경의 상세정보 나옴
 ```bash
 (base) C:\Users\student>conda info

     active environment : base
    active env location : C:\Users\student\Anaconda3
            shell level : 1
       user config file : C:\Users\student\.condarc
 populated config files :
          conda version : 4.7.12
    conda-build version : 3.18.9
         python version : 3.7.4.final.0
       virtual packages :
       base environment : C:\Users\student\Anaconda3  (writable)
           channel URLs : https://repo.anaconda.com/pkgs/main/win-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/win-64
                          https://repo.anaconda.com/pkgs/r/noarch
                          https://repo.anaconda.com/pkgs/msys2/win-64
                          https://repo.anaconda.com/pkgs/msys2/noarch
          package cache : C:\Users\student\Anaconda3\pkgs
                          C:\Users\student\.conda\pkgs
                          C:\Users\student\AppData\Local\conda\conda\pkgs
       envs directories : C:\Users\student\Anaconda3\envs
                          C:\Users\student\.conda\envs
                          C:\Users\student\AppData\Local\conda\conda\envs
               platform : win-64
             user-agent : conda/4.7.12 requests/2.22.0 CPython/3.7.4 Windows/10 Windows/10.0.18362
          administrator : False
             netrc file : None
           offline mode : False


(base) C:\Users\student>
 ```

### 4. 가상환경 실행
 - conda activate python374
```bash
(base) C:\Users\student>conda activate python374

(python374) C:\Users\student>
```

#### 5. 가상환경 종료
 - conda deactivate
```bash
(python374) C:\Users\student>conda deactivate

(base) C:\Users\student>
```

#### 6. 가상환경 캐시삭제
 - conda clean --all
	- 아나콘다의 clean 명령어를 통해서 캐시를 삭제할 수 있다. 인덱스 캐시, 잠긴 파일, 사용하지 않는 패키지, 소스 캐시 등을 삭제할 수 있다.

#### 7. 모듈관련 설치
 - conda install [모듈이름]
```bash
activate 명령어를 통해 원하는 가상환경에 접속해서 아래와 같이 입력한다.
(base) PS C:\Users\myounghwan> conda install scikit-learn scipy
WARNING: The conda.compat module is deprecated and will be removed in a future release.
Collecting package metadata: done
Solving environment: done

## Package Plan ##

  environment location: C:\Users\myounghwan\Anaconda3

  added / updated specs:
    - scikit-learn
    - scipy


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    conda-4.8.0                |           py37_1         3.1 MB
    conda-package-handling-1.3.11|           py37_0         280 KB
    ------------------------------------------------------------
                                           Total:         3.3 MB

The following NEW packages will be INSTALLED:

  conda-package-han~ pkgs/main/win-64::conda-package-handling-1.3.11-py37_0

The following packages will be UPDATED:

  conda                                       4.6.11-py37_0 --> 4.8.0-py37_1


Proceed ([y]/n)?
이런식으로 설치하며 된다.
```
