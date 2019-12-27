# 12월 27일 web scraping review

#### urllib

- url작업을 위한 여러 모듈을 모은 패키지(python3)

#### urllib.request

- url주소의 문서를 열고 일기 위해 사용하는 모듈 (예)urlopen(url).read() 읽은 데이터가 python실행환경의 메모리로 로드

#### urllib.parser

 - url주소의 문서를 구문 분석을 위해 사용
    응답코드
  - 200 OK (요청 성공) 
   
   - 404 Not Found (4xx 클라이언트 에러) 
   
   - 500 Server Error (4xx 서버 에러)
   
 - urllib.error
    - urllib.request에 의해 발생되는 예외를 포함하고 있는 모듈
 - urllib.parser.urlencode(파라미터)
    - url요청 파라미터를 인코딩 함수
 - urllib.request.urlretrieved(url,  로컬에 저장될 파일 이름)
    - url주소의 문서를 다운로드해서 로컬 저장



#### requests 모듈

 - 간편하게 HTTP request 보내고, 세션을 유지하기 용이하며,  파라미터 데이터들을 편리하게 인코딩 기능을 제공 모듈

- get(), post(), head(), put(), delete()

get(url, headers=, cookies=) 함수로부터 반환 객체 - Response객체

Response객체.request - 요청을 보낸 객체(request객체)에 접근
Response객체.status_code  - 요청에 대한 응답 코드
Response객체.text - 요청에 대한 응답 body내용을 text 변환
Response객체.content - 요청에 대한 응답 body내용을 byte 변환
Response객체.raise_for_status() - 에러 발생(200 OK 응답이 아닌 경우)
Response객체.json()  - 요청에 대한 응답이 JSON이 경우 딕셔너리 형태로 변환

BeautifulSoup - HTML, XML 파싱하는데 사용되는 라이브러리
BeautifulSoup( , 파서) 생성자 함수에 파싱할 문서 객체(string, url)를 인수로 전달해서 파싱 객체를 생성
html.parser - 단순한  html문서 구문분석 파서
lxml - html, xml에 대해서 더 유연하고 빠르게 구문분석 파서
xml - xml문서만 구문분석 파서
html5lib - 구조가 복잡한 html 문서 구문분석 파서
bs.부모노드.자식노드.자식노드. ... 방식으로 계층 구조로 구성된 html문서내에 특정 요소를 접근
bs.find_all(태그명 , {속성명: 값,.....})    - 태그명, 태그와 속성면과 값 등등의 조건에 맞는 모든 태그를 반환하는 함수
bs.find(태그명 , {속성명: 값,.....}) - 태그명, 태그와 속성면과 값 등등의 조건에 맞는 하나의 태그를 반환하는 함수
find()등의 함수로부터 반환되는 객체 타입 bs.element.Tag
bs.element.Tag.attrs 속성으로부터 태그의 모든 속성 추출
[ 태그의 내용 추출]
bs.element.Tag.string, 
bs.element.Tag.text, 
bs.element.Tag.contents, 
bs.element.Tag.strings
bs.element.Tag.stripped_strings 
bs.element.Tag.get_text()

bs.element.Tag.name  - 태그명 추출
bs.element.Tag.parent - 태그의 부모 노드 추출
bs.element.Tag.child - 태그의 자식 노드 추출
bs.element.Tag.

bs.elect() - CSS의 선택자를 조건에 맞는 모든 태그를 반환하는 함수

모든 요소 CSS의 선택자 : *
`<p id="p1"></> 태그의  id 속성  선택자`  : #p1
`<p class="p1"></> 태그의  class 속성  선택자`  : .p1
`<input type="number"></>  태그의  type 속성이 number 선택자` : input[type='number']
`<input type="text|password|button|radio|checkbox|reset|submit|phone|url|color|..."></>`

